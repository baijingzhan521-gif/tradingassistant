from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.data.ohlcv_service import OhlcvService
from app.indicators.atr import apply_atr_indicator
from app.indicators.candle_profile import summarize_candle_profile
from app.indicators.divergence import apply_divergence_indicator, divergence_profile_from_row, empty_divergence_profile
from app.indicators.ema import apply_ema_indicators
from app.indicators.market_structure import (
    classify_structure,
    compute_trend_strength,
    determine_ema_alignment,
    determine_trend_bias,
)
from app.indicators.swings import identify_swings
from app.schemas.analysis import TimeframeAnalysis
from app.schemas.common import Action, Bias, RecommendedTiming, StructureState, TriggerState, VolatilityState
from app.services.strategy_service import StrategyService
from app.strategies.scoring import ScoreCard
from app.strategies.windowed_mtf import PreparedTimeframe, WindowedMTFStrategy
from app.utils.math_utils import pct_distance
from app.utils.timeframes import TIMEFRAME_TO_MINUTES, get_strategy_required_timeframes


logger = logging.getLogger(__name__)
BACKTEST_SNAPSHOT_TAIL_BARS = 31
POSITION_MAP_COLUMNS = (
    "band_upper",
    "band_lower",
    "band_volatility_unit",
    "axis_distance_vol",
    "ema55_distance_vol",
    "band_position",
)


@dataclass
class BacktestAssumptions:
    exit_profile: str = "scaled_1r_to_2r"
    take_profit_mode: str = "scaled"
    fixed_take_profit_r: Optional[float] = None
    scaled_tp1_r: Optional[float] = None
    scaled_tp2_r: Optional[float] = None
    taker_fee_bps: float = 5.0
    slippage_bps: float = 2.0
    intraday_max_hold_bars: int = 240
    swing_max_hold_bars: int = 240
    tp1_scale_out: float = 0.5
    move_stop_to_entry_after_tp1: bool = True
    conservative_same_bar_exit: bool = True
    lookback: int = 300
    cache_dir: str = "artifacts/backtests/cache"
    long_exit: Optional[dict[str, Any]] = None
    short_exit: Optional[dict[str, Any]] = None
    swing_detection_mode: str = "centered"
    # --- trailing stop ---
    trailing_stop_enabled: bool = False
    trailing_stop_atr_mult: float = 3.0
    trailing_stop_activation_r: float = 1.0
    # --- leverage ---
    leverage: float = 1.0
    max_leverage: float = 3.0
    # --- drawdown circuit-breaker ---
    drawdown_circuit_breaker_enabled: bool = False
    drawdown_level1_pct: float = 10.0
    drawdown_level2_pct: float = 20.0
    drawdown_level3_pct: float = 25.0
    drawdown_cooldown_bars: int = 168


@dataclass
class BacktestTrade:
    symbol: str
    strategy_profile: str
    side: str
    higher_bias: str
    trend_strength: int
    signal_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    bars_held: int
    exit_reason: str
    confidence: int
    tp1_hit: bool
    tp2_hit: bool
    pnl_pct: float
    pnl_r: float
    gross_pnl_quote: float
    fees_quote: float
    leverage: float = 1.0


@dataclass
class BacktestSummary:
    strategy_profile: str
    symbol: Optional[str]
    total_trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    payoff_ratio: float
    profit_factor: float
    expectancy_r: float
    avg_r: float
    median_r: float
    cumulative_r: float
    cumulative_return_pct: float
    max_drawdown_r: float
    avg_holding_bars: float
    avg_holding_hours: float
    tp1_hit_rate: float
    tp2_hit_rate: float
    signals_now: int
    skipped_entries: int


@dataclass
class BacktestReport:
    generated_at: str
    exchange: str
    market_type: str
    start: str
    end: str
    symbols: list[str]
    strategy_profiles: list[str]
    assumptions: dict[str, Any]
    overall: list[BacktestSummary]
    by_symbol: list[BacktestSummary]
    trades: list[BacktestTrade] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "start": self.start,
            "end": self.end,
            "symbols": self.symbols,
            "strategy_profiles": self.strategy_profiles,
            "assumptions": self.assumptions,
            "overall": [asdict(item) for item in self.overall],
            "by_symbol": [asdict(item) for item in self.by_symbol],
            "trades": [asdict(item) for item in self.trades],
        }


@dataclass
class _SignalSnapshot:
    action: Action
    bias: Bias
    trend_strength: int
    confidence: int
    recommended_timing: RecommendedTiming
    entry_zone_low: Optional[float]
    entry_zone_high: Optional[float]
    stop_price: Optional[float]
    tp1_price: Optional[float]
    tp2_price: Optional[float]
    invalidation_price: Optional[float]
    timestamp: datetime


@dataclass
class _PendingEntry:
    signal: _SignalSnapshot


@dataclass
class _OpenPosition:
    symbol: str
    strategy_profile: str
    side: Action
    higher_bias: Bias
    trend_strength: int
    signal_time: datetime
    entry_time: datetime
    entry_price: float
    initial_stop_price: float
    current_stop_price: float
    tp1_price: float
    tp2_price: float
    take_profit_mode: str
    fixed_take_profit_r: Optional[float]
    confidence: int
    tp1_scale_out: float = 0.5
    move_stop_to_entry_after_tp1: bool = True
    max_hold_bars: Optional[int] = None
    remaining_qty: float = 1.0
    realized_pnl_quote: float = 0.0
    fees_quote: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    bars_held: int = 0
    last_fill_price: float = 0.0
    # --- trailing stop ---
    trailing_stop_enabled: bool = False
    trailing_stop_atr_mult: float = 3.0
    trailing_stop_activation_r: float = 1.0
    trailing_stop_active: bool = False
    highest_price_since_entry: float = 0.0
    lowest_price_since_entry: float = float("inf")
    # --- leverage ---
    leverage: float = 1.0


class BacktestService:
    def __init__(
        self,
        *,
        ohlcv_service: OhlcvService,
        strategy_service: StrategyService,
        assumptions: Optional[BacktestAssumptions] = None,
    ) -> None:
        self.ohlcv_service = ohlcv_service
        self.strategy_service = strategy_service
        self.assumptions = assumptions or BacktestAssumptions()
        self.cache_dir = Path(self.assumptions.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._validate_assumptions()

    def run(
        self,
        *,
        exchange: str,
        market_type: str,
        symbols: list[str],
        strategy_profiles: list[str],
        start: datetime,
        end: datetime,
    ) -> BacktestReport:
        by_symbol: list[BacktestSummary] = []
        trades: list[BacktestTrade] = []

        for symbol in symbols:
            for strategy_profile in strategy_profiles:
                summary, strategy_trades = self._run_symbol_strategy(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    strategy_profile=strategy_profile,
                    start=start,
                    end=end,
                )
                by_symbol.append(summary)
                trades.extend(strategy_trades)

        overall = []
        for strategy_profile in strategy_profiles:
            strategy_trades = [trade for trade in trades if trade.strategy_profile == strategy_profile]
            signals_now = sum(item.signals_now for item in by_symbol if item.strategy_profile == strategy_profile)
            skipped_entries = sum(item.skipped_entries for item in by_symbol if item.strategy_profile == strategy_profile)
            overall.append(
                self._summarize_trades(
                    trades=strategy_trades,
                    strategy_profile=strategy_profile,
                    symbol=None,
                    signals_now=signals_now,
                    skipped_entries=skipped_entries,
                )
            )

        return BacktestReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            exchange=exchange,
            market_type=market_type,
            start=start.isoformat(),
            end=end.isoformat(),
            symbols=symbols,
            strategy_profiles=strategy_profiles,
            assumptions=asdict(self.assumptions),
            overall=overall,
            by_symbol=by_symbol,
            trades=trades,
        )

    def prepare_history(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        strategy_profile: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        return self._load_history(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            strategy_profile=strategy_profile,
            start=start,
            end=end,
        )

    def prepare_enriched_history(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        strategy_profile: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        frames = self.prepare_history(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            strategy_profile=strategy_profile,
            start=start,
            end=end,
        )
        return self._prepare_enriched_frames(strategy_profile=strategy_profile, frames=frames)

    def run_symbol_strategy_with_frames(
        self,
        *,
        symbol: str,
        strategy_profile: str,
        start: datetime,
        end: datetime,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[BacktestSummary, list[BacktestTrade]]:
        return self._run_symbol_strategy_with_frames(
            symbol=symbol,
            strategy_profile=strategy_profile,
            start=start,
            end=end,
            frames=frames,
        )

    def run_symbol_strategy_with_enriched_frames(
        self,
        *,
        symbol: str,
        strategy_profile: str,
        start: datetime,
        end: datetime,
        enriched_frames: dict[str, pd.DataFrame],
    ) -> tuple[BacktestSummary, list[BacktestTrade]]:
        return self._run_symbol_strategy_on_enriched_frames(
            symbol=symbol,
            strategy_profile=strategy_profile,
            start=start,
            end=end,
            enriched=enriched_frames,
        )

    def _run_symbol_strategy(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        strategy_profile: str,
        start: datetime,
        end: datetime,
    ) -> tuple[BacktestSummary, list[BacktestTrade]]:
        frames = self._load_history(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            strategy_profile=strategy_profile,
            start=start,
            end=end,
        )
        return self._run_symbol_strategy_with_frames(
            symbol=symbol,
            strategy_profile=strategy_profile,
            start=start,
            end=end,
            frames=frames,
        )

    def _prepare_enriched_frames(
        self,
        strategy_profile: str,
        frames: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        strategy = self.strategy_service.build_strategy(strategy_profile)
        if not isinstance(strategy, WindowedMTFStrategy):
            raise TypeError(f"Backtest currently supports WindowedMTFStrategy only, got {strategy_profile}")
        return {
            timeframe: self._enrich_frame(strategy, timeframe, frame)
            for timeframe, frame in frames.items()
        }

    def _run_symbol_strategy_with_frames(
        self,
        *,
        symbol: str,
        strategy_profile: str,
        start: datetime,
        end: datetime,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[BacktestSummary, list[BacktestTrade]]:
        enriched = self._prepare_enriched_frames(strategy_profile, frames)
        return self._run_symbol_strategy_on_enriched_frames(
            symbol=symbol,
            strategy_profile=strategy_profile,
            start=start,
            end=end,
            enriched=enriched,
        )

    def _run_symbol_strategy_on_enriched_frames(
        self,
        *,
        symbol: str,
        strategy_profile: str,
        start: datetime,
        end: datetime,
        enriched: dict[str, pd.DataFrame],
    ) -> tuple[BacktestSummary, list[BacktestTrade]]:
        strategy = self.strategy_service.build_strategy(strategy_profile)
        if not isinstance(strategy, WindowedMTFStrategy):
            raise TypeError(f"Backtest currently supports WindowedMTFStrategy only, got {strategy_profile}")

        trigger_tf = str(strategy.window_config["trigger_timeframe"])
        required = tuple(get_strategy_required_timeframes(strategy_profile))
        trigger_frame = enriched[trigger_tf]
        trigger_end_idx = int(trigger_frame["timestamp"].searchsorted(pd.Timestamp(end), side="left"))
        indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}

        trades: list[BacktestTrade] = []
        pending_entry: Optional[_PendingEntry] = None
        position: Optional[_OpenPosition] = None
        snapshot_cache: dict[str, tuple[int, PreparedTimeframe]] = {}
        signals_now = 0
        skipped_entries = 0
        cooldown_remaining = 0
        cooldown_bars_after_exit = int(strategy.config.get("backtest", {}).get("cooldown_bars_after_exit", 0))

        for trigger_idx in range(trigger_end_idx):
            candle = trigger_frame.iloc[trigger_idx]
            ts = candle["timestamp"].to_pydatetime()

            if pending_entry is not None:
                maybe_position = self._open_pending_entry(
                    symbol=symbol,
                    strategy_profile=strategy_profile,
                    pending_entry=pending_entry,
                    candle=candle,
                )
                if maybe_position is None:
                    skipped_entries += 1
                else:
                    position = maybe_position
                pending_entry = None

            if position is not None:
                trade = self._update_open_position(
                    position=position,
                    candle=candle,
                    max_hold_bars=self._max_hold_bars(strategy_profile),
                )
                if trade is not None:
                    trades.append(trade)
                    position = None
                    cooldown_remaining = cooldown_bars_after_exit

            if ts < start:
                continue

            current_indices: dict[str, int] = {trigger_tf: trigger_idx}
            ready = True
            for timeframe in required:
                if timeframe == trigger_tf:
                    continue
                frame = enriched[timeframe]
                pointer = indices[timeframe]
                while pointer + 1 < len(frame) and frame.iloc[pointer + 1]["timestamp"] <= candle["timestamp"]:
                    pointer += 1
                indices[timeframe] = pointer
                if frame.iloc[pointer]["timestamp"] > candle["timestamp"]:
                    ready = False
                    break
                current_indices[timeframe] = pointer
            if not ready:
                continue

            min_required = max(int(self.assumptions.lookback // 3), 20)
            if any(index < min_required for index in current_indices.values()):
                continue

            if position is not None or pending_entry is not None:
                continue
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            signal = self._evaluate_signal(
                strategy=strategy,
                strategy_profile=strategy_profile,
                enriched=enriched,
                indices=current_indices,
                timestamp=ts,
                snapshot_cache=snapshot_cache,
            )
            if signal.action in {Action.LONG, Action.SHORT} and signal.recommended_timing == RecommendedTiming.NOW:
                signals_now += 1
                pending_entry = _PendingEntry(signal=signal)

        if position is not None and trigger_end_idx > 0:
            final_candle = trigger_frame.iloc[trigger_end_idx - 1]
            trades.append(self._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"])))

        summary = self._summarize_trades(
            trades=trades,
            strategy_profile=strategy_profile,
            symbol=symbol,
            signals_now=signals_now,
            skipped_entries=skipped_entries,
        )
        return summary, trades

    def _validate_assumptions(self) -> None:
        if self.assumptions.take_profit_mode not in {"scaled", "fixed_r"}:
            raise ValueError(f"Unsupported take_profit_mode: {self.assumptions.take_profit_mode}")
        if self.assumptions.swing_detection_mode not in {"centered", "confirmed"}:
            raise ValueError("swing_detection_mode must be 'centered' or 'confirmed'")
        if self.assumptions.take_profit_mode == "fixed_r":
            if self.assumptions.fixed_take_profit_r is None or self.assumptions.fixed_take_profit_r <= 0:
                raise ValueError("fixed_take_profit_r must be > 0 when take_profit_mode='fixed_r'")
        if self.assumptions.scaled_tp1_r is not None and self.assumptions.scaled_tp1_r <= 0:
            raise ValueError("scaled_tp1_r must be > 0 when provided")
        if self.assumptions.scaled_tp2_r is not None and self.assumptions.scaled_tp2_r <= 0:
            raise ValueError("scaled_tp2_r must be > 0 when provided")
        self._validate_side_exit_config(self.assumptions.long_exit, label="long_exit")
        self._validate_side_exit_config(self.assumptions.short_exit, label="short_exit")

    @staticmethod
    def _validate_side_exit_config(config: Optional[dict[str, Any]], *, label: str) -> None:
        if not config:
            return
        mode = str(config.get("take_profit_mode", "scaled"))
        if mode not in {"scaled", "fixed_r"}:
            raise ValueError(f"{label}.take_profit_mode must be 'scaled' or 'fixed_r'")
        fixed_take_profit_r = config.get("fixed_take_profit_r")
        if mode == "fixed_r" and (fixed_take_profit_r is None or float(fixed_take_profit_r) <= 0):
            raise ValueError(f"{label}.fixed_take_profit_r must be > 0 when take_profit_mode='fixed_r'")
        for field_name in ("scaled_tp1_r", "scaled_tp2_r", "tp1_scale_out", "max_hold_bars"):
            value = config.get(field_name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{label}.{field_name} must be > 0 when provided")

    def _default_max_hold_bars(self, strategy_profile: str) -> int:
        if strategy_profile == "intraday_mtf_v1":
            return int(self.assumptions.intraday_max_hold_bars)
        return int(self.assumptions.swing_max_hold_bars)

    def _resolve_exit_config(self, *, side: Action, strategy_profile: str) -> dict[str, Any]:
        default_config: dict[str, Any] = {
            "take_profit_mode": self.assumptions.take_profit_mode,
            "fixed_take_profit_r": self.assumptions.fixed_take_profit_r,
            "scaled_tp1_r": self.assumptions.scaled_tp1_r,
            "scaled_tp2_r": self.assumptions.scaled_tp2_r,
            "tp1_scale_out": self.assumptions.tp1_scale_out,
            "move_stop_to_entry_after_tp1": self.assumptions.move_stop_to_entry_after_tp1,
            "max_hold_bars": self._default_max_hold_bars(strategy_profile),
        }
        side_override = self.assumptions.long_exit if side == Action.LONG else self.assumptions.short_exit
        if side_override:
            default_config.update(side_override)
        return default_config

    def _evaluate_signal(
        self,
        *,
        strategy: WindowedMTFStrategy,
        strategy_profile: str,
        enriched: dict[str, pd.DataFrame],
        indices: dict[str, int],
        timestamp: datetime,
        snapshot_cache: Optional[dict[str, tuple[int, PreparedTimeframe]]] = None,
    ) -> _SignalSnapshot:
        prepared: dict[str, PreparedTimeframe] = {}
        for timeframe, idx in indices.items():
            cached = snapshot_cache.get(timeframe) if snapshot_cache is not None else None
            if cached is not None and cached[0] == idx:
                prepared[timeframe] = cached[1]
                continue
            snapshot = self._build_snapshot(strategy, timeframe, enriched[timeframe], idx)
            prepared[timeframe] = snapshot
            if snapshot_cache is not None:
                snapshot_cache[timeframe] = (idx, snapshot)

        higher_bias, trend_strength = strategy._derive_higher_timeframe_bias(prepared)
        setup_key = str(strategy.window_config["setup_timeframe"])
        trigger_key = str(strategy.window_config["trigger_timeframe"])
        reference_key = str(strategy.window_config.get("reference_timeframe", setup_key))

        volatility_state = strategy._derive_volatility_state(prepared[setup_key])
        is_trend_friendly = strategy._is_trend_friendly(
            higher_bias=higher_bias,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
        )

        scorecard = ScoreCard(base=50)
        higher_label = strategy._format_timeframe_group(tuple(strategy.window_config["higher_timeframes"]))
        if higher_bias == Bias.BULLISH:
            scorecard.add(15, "higher_bias", f"{higher_label} 同步偏多")
        elif higher_bias == Bias.BEARISH:
            scorecard.add(15, "higher_bias", f"{higher_label} 同步偏空")
        else:
            scorecard.add(-20, "higher_conflict", f"{higher_label} 没有形成同向共振")

        setup_assessment = strategy._assess_setup(
            higher_bias,
            prepared[setup_key],
            setup_key,
            reference_ctx=prepared[reference_key],
            current_price=prepared[trigger_key].model.close,
        )
        scorecard.add(setup_assessment["score"], f"{setup_key}_setup", setup_assessment["score_note"])

        trigger_assessment = strategy._assess_trigger(
            higher_bias,
            prepared[trigger_key],
            trigger_key,
            trend_strength=trend_strength,
        )
        scorecard.add(trigger_assessment["score"], f"{trigger_key}_trigger", trigger_assessment["score_note"])

        if volatility_state == VolatilityState.HIGH:
            scorecard.add(-15, "volatility", f"{setup_key} ATR 百分比偏高")
        elif volatility_state == VolatilityState.LOW:
            scorecard.add(3, "volatility", f"{setup_key} 波动可控")

        confidence = scorecard.total
        action, bias, recommended_timing = strategy._decide(
            higher_bias=higher_bias,
            trend_friendly=is_trend_friendly,
            setup_assessment=setup_assessment,
            trigger_assessment=trigger_assessment,
            confidence=confidence,
        )
        trade_plan = strategy._build_trade_plan(
            action=action,
            bias=bias,
            setup_ctx=prepared[setup_key],
            reference_ctx=prepared[reference_key],
            current_price=prepared[trigger_key].model.close,
            setup_key=setup_key,
            reference_key=reference_key,
        )

        return _SignalSnapshot(
            action=action,
            bias=bias,
            trend_strength=trend_strength,
            confidence=confidence,
            recommended_timing=recommended_timing,
            entry_zone_low=float(trade_plan["entry_zone"].low) if trade_plan["entry_zone"] else None,
            entry_zone_high=float(trade_plan["entry_zone"].high) if trade_plan["entry_zone"] else None,
            stop_price=float(trade_plan["stop_loss"].price) if trade_plan["stop_loss"] else None,
            tp1_price=float(trade_plan["take_profit_hint"].tp1) if trade_plan["take_profit_hint"] else None,
            tp2_price=float(trade_plan["take_profit_hint"].tp2) if trade_plan["take_profit_hint"] else None,
            invalidation_price=float(trade_plan["invalidation_price"]) if trade_plan["invalidation_price"] is not None else None,
            timestamp=timestamp,
        )

    def _load_history(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        strategy_profile: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        timeframes = tuple(get_strategy_required_timeframes(strategy_profile))
        preload_minutes = max(TIMEFRAME_TO_MINUTES[item] * self.assumptions.lookback for item in timeframes)
        padded_start = start - timedelta(minutes=preload_minutes)

        frames: dict[str, pd.DataFrame] = {}
        for timeframe in timeframes:
            cache_path = self._cache_path(symbol=symbol, timeframe=timeframe, start=padded_start, end=end)
            if cache_path.exists():
                frame = pd.read_csv(cache_path)
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
                frames[timeframe] = frame
                continue

            frame = self.ohlcv_service.fetch_ohlcv_range(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                start=padded_start,
                end=end,
            )
            frame.to_csv(cache_path, index=False)
            frames[timeframe] = frame
        return frames

    def _enrich_frame(self, strategy: WindowedMTFStrategy, timeframe: str, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy().sort_values("timestamp").reset_index(drop=True)
        ema_periods = tuple(sorted(set(int(period) for period in strategy.config["ema"]["periods"])))
        enriched = apply_ema_indicators(enriched, periods=ema_periods)
        enriched = apply_atr_indicator(enriched, period=int(strategy.config["atr_period"]))
        enriched = identify_swings(
            enriched,
            window=int(strategy.config["swing_window"]),
            mode=str(self.assumptions.swing_detection_mode),
        )

        enriched["ema21"] = enriched["ema_21"].astype(float)
        enriched["ema55"] = enriched["ema_55"].astype(float)
        enriched["ema100"] = enriched["ema_100"].astype(float)
        enriched["ema200"] = enriched["ema_200"].astype(float)
        enriched["atr14"] = enriched[f"atr_{int(strategy.config['atr_period'])}"].astype(float)
        enriched["atr_pct"] = np.where(enriched["close"] != 0, (enriched["atr14"] / enriched["close"]) * 100, 0.0)

        bull_align = (
            (enriched["ema21"] > enriched["ema55"])
            & (enriched["ema55"] > enriched["ema100"])
            & (enriched["ema100"] > enriched["ema200"])
        )
        bear_align = (
            (enriched["ema21"] < enriched["ema55"])
            & (enriched["ema55"] < enriched["ema100"])
            & (enriched["ema100"] < enriched["ema200"])
        )
        enriched["ema_alignment"] = np.select(
            [bull_align, bear_align],
            [Bias.BULLISH.value, Bias.BEARISH.value],
            default="mixed",
        )
        enriched["trend_bias"] = np.select(
            [
                (enriched["close"] > enriched["ema200"]) & bull_align,
                (enriched["close"] < enriched["ema200"]) & bear_align,
            ],
            [Bias.BULLISH.value, Bias.BEARISH.value],
            default=Bias.NEUTRAL.value,
        )

        recent_highs: deque[float] = deque(maxlen=3)
        recent_lows: deque[float] = deque(maxlen=3)
        last_swing_high: list[Optional[float]] = []
        last_swing_low: list[Optional[float]] = []
        structure_states: list[str] = []

        for row in enriched.itertuples(index=False):
            swing_high_marker = getattr(row, "swing_high_marker")
            swing_low_marker = getattr(row, "swing_low_marker")
            if pd.notna(swing_high_marker):
                recent_highs.append(float(swing_high_marker))
            if pd.notna(swing_low_marker):
                recent_lows.append(float(swing_low_marker))

            last_swing_high.append(recent_highs[-1] if recent_highs else None)
            last_swing_low.append(recent_lows[-1] if recent_lows else None)
            structure_states.append(classify_structure(list(recent_highs), list(recent_lows)).value)

        enriched["recent_swing_high"] = last_swing_high
        enriched["recent_swing_low"] = last_swing_low
        enriched["structure_state"] = structure_states

        history_window = max(int(strategy.config["micro"]["confirmation_lookback"]), 3) - 1
        doji_body_ratio_max = float(strategy.config["micro"].get("doji_body_ratio_max", 0.12))
        reversal_wick_ratio_min = float(strategy.config["micro"].get("reversal_wick_ratio_min", 0.2))
        candle_range = (enriched["high"] - enriched["low"]).clip(lower=0.0)
        body = (enriched["close"] - enriched["open"]).abs()
        upper_wick = (enriched["high"] - enriched[["open", "close"]].max(axis=1)).clip(lower=0.0)
        lower_wick = (enriched[["open", "close"]].min(axis=1) - enriched["low"]).clip(lower=0.0)
        quote_volume = enriched["volume"] * enriched["close"]
        range_median = candle_range.shift(1).rolling(history_window, min_periods=1).median().fillna(candle_range)
        volume_median = enriched["volume"].shift(1).rolling(history_window, min_periods=1).median().fillna(enriched["volume"])
        quote_median = quote_volume.shift(1).rolling(history_window, min_periods=1).median().fillna(quote_volume)

        safe_range = candle_range.replace(0.0, np.nan)
        enriched["cp_body_ratio"] = (body / safe_range).fillna(0.0)
        enriched["cp_upper_wick_ratio"] = (upper_wick / safe_range).fillna(0.0)
        enriched["cp_lower_wick_ratio"] = (lower_wick / safe_range).fillna(0.0)
        enriched["cp_range_ratio"] = np.where(range_median != 0, candle_range / range_median, 0.0)
        enriched["cp_volume_ratio"] = np.where(volume_median != 0, enriched["volume"] / volume_median, 0.0)
        enriched["cp_quote_volume_ratio"] = np.where(quote_median != 0, quote_volume / quote_median, 0.0)
        enriched["cp_is_volume_contracting"] = (enriched["cp_volume_ratio"] <= 0.9) | (enriched["cp_quote_volume_ratio"] <= 0.9)
        enriched["cp_is_spiky"] = (enriched["cp_range_ratio"] >= 1.6) | (
            enriched[["cp_upper_wick_ratio", "cp_lower_wick_ratio"]].max(axis=1) >= 0.45
        )
        enriched["cp_is_doji"] = enriched["cp_body_ratio"] <= doji_body_ratio_max
        long_lower_wick_reversal = (
            (enriched["cp_lower_wick_ratio"] >= reversal_wick_ratio_min)
            & (enriched["cp_lower_wick_ratio"] > enriched["cp_upper_wick_ratio"])
        )
        long_upper_wick_reversal = (
            (enriched["cp_upper_wick_ratio"] >= reversal_wick_ratio_min)
            & (enriched["cp_upper_wick_ratio"] > enriched["cp_lower_wick_ratio"])
        )
        enriched["cp_has_bullish_rejection"] = (
            (enriched["close"] >= enriched["open"])
            & long_lower_wick_reversal
        )
        enriched["cp_has_bearish_rejection"] = (
            (enriched["close"] <= enriched["open"])
            & long_upper_wick_reversal
        )
        enriched["cp_has_bullish_reversal_candle"] = enriched["cp_has_bullish_rejection"] | (
            enriched["cp_is_doji"] & long_lower_wick_reversal
        )
        enriched["cp_has_bearish_reversal_candle"] = enriched["cp_has_bearish_rejection"] | (
            enriched["cp_is_doji"] & long_upper_wick_reversal
        )
        enriched["cp_latest_quote_volume"] = quote_volume
        enriched["cp_median_quote_volume"] = quote_median

        enriched["value_zone_low"] = enriched[["ema21", "ema55", "ema100"]].min(axis=1)
        enriched["value_zone_high"] = enriched[["ema21", "ema55", "ema100"]].max(axis=1)
        enriched["execution_zone_low"] = enriched[["ema21", "ema55"]].min(axis=1)
        enriched["execution_zone_high"] = enriched[["ema21", "ema55"]].max(axis=1)
        enriched["distance_to_execution_atr"] = np.where(
            enriched["close"] < enriched["execution_zone_low"],
            (enriched["execution_zone_low"] - enriched["close"]) / enriched["atr14"].replace(0.0, np.nan),
            np.where(
                enriched["close"] > enriched["execution_zone_high"],
                (enriched["close"] - enriched["execution_zone_high"]) / enriched["atr14"].replace(0.0, np.nan),
                0.0,
            ),
        )
        enriched["distance_to_execution_atr"] = enriched["distance_to_execution_atr"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        enriched["is_pullback_to_value_area"] = (
            enriched["distance_to_execution_atr"] <= float(strategy.config["execution"]["pullback_distance_atr"])
        )
        enriched["is_extended"] = (
            enriched["distance_to_execution_atr"] >= float(strategy.config["execution"]["extension_distance_atr"])
        )
        position_config = dict(strategy.config.get("position_map", {}))
        volatility_mode = str(position_config.get("band_volatility_mode", "atr")).lower()
        atr_band_mult = float(position_config.get("band_atr_mult", 1.5))
        if volatility_mode == "std":
            std_period = int(position_config.get("band_std_period", 20))
            std_mult = float(position_config.get("band_std_mult", 2.0))
            rolling_std = enriched["close"].rolling(std_period).std(ddof=0)
            band_volatility_unit = rolling_std.where(rolling_std > 0, enriched["atr14"]).astype(float)
            band_half_width = band_volatility_unit * std_mult
        else:
            band_volatility_unit = enriched["atr14"].astype(float)
            band_half_width = band_volatility_unit * atr_band_mult
        enriched["band_volatility_unit"] = band_volatility_unit
        enriched["band_upper"] = enriched["ema100"] + band_half_width
        enriched["band_lower"] = enriched["ema100"] - band_half_width
        safe_band_unit = enriched["band_volatility_unit"].replace(0.0, np.nan)
        enriched["axis_distance_vol"] = (
            ((enriched["close"] - enriched["ema100"]) / safe_band_unit)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        enriched["ema55_distance_vol"] = (
            ((enriched["close"] - enriched["ema55"]) / safe_band_unit)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        safe_band_width = (enriched["band_upper"] - enriched["band_lower"]).replace(0.0, np.nan)
        enriched["band_position"] = (
            ((enriched["close"] - enriched["band_lower"]) / safe_band_width)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.5)
        )

        trend_scores: list[int] = []
        for idx in range(len(enriched)):
            row = enriched.iloc[idx]
            tail = enriched.iloc[max(0, idx - 4) : idx + 1]
            trend_scores.append(
                compute_trend_strength(
                    tail,
                    close=float(row["close"]),
                    ema21=float(row["ema21"]),
                    ema55=float(row["ema55"]),
                    ema100=float(row["ema100"]),
                    ema200=float(row["ema200"]),
                    alignment=determine_ema_alignment(
                        float(row["ema21"]),
                        float(row["ema55"]),
                        float(row["ema100"]),
                        float(row["ema200"]),
                    ),
                    bias=determine_trend_bias(
                        float(row["close"]),
                        float(row["ema200"]),
                        determine_ema_alignment(
                            float(row["ema21"]),
                            float(row["ema55"]),
                            float(row["ema100"]),
                            float(row["ema200"]),
                        ),
                    ),
                    structure_state=StructureState(str(row["structure_state"])),
                )
            )
        enriched["trend_score"] = trend_scores
        divergence_config = dict(strategy.config.get("divergence", {}))
        if bool(divergence_config.get("enabled", False)):
            enriched = apply_divergence_indicator(
                enriched,
                rsi_period=int(divergence_config.get("rsi_period", 10)),
                swing_window=int(divergence_config.get("swing_window", 14)),
                ema_period=int(divergence_config.get("ema_period", 34)),
                atr_period=int(strategy.config["atr_period"]),
                min_rsi_diff=float(divergence_config.get("min_rsi_diff", 1.6)),
                min_move_atr_mult=float(divergence_config.get("min_move_atr_mult", 0.3)),
                stretch_atr_mult=float(divergence_config.get("stretch_atr_mult", 0.8)),
                wick_ratio_min=float(divergence_config.get("wick_ratio_min", 0.4)),
                min_reversal_score=int(divergence_config.get("min_reversal_score", 2)),
                cooldown_bars=int(divergence_config.get("cooldown_bars", 20)),
            )
        return enriched

    def _build_snapshot(
        self,
        strategy: WindowedMTFStrategy,
        timeframe: str,
        enriched: pd.DataFrame,
        idx: int,
    ) -> PreparedTimeframe:
        row = enriched.iloc[idx]
        history_start = max(0, idx - BACKTEST_SNAPSHOT_TAIL_BARS + 1)
        history_df = enriched.iloc[history_start : idx + 1]
        close = float(row["close"])
        atr14 = float(row["atr14"])
        ema55 = float(row["ema55"])
        ema100 = float(row["ema100"])
        swing_high = None if pd.isna(row["recent_swing_high"]) else float(row["recent_swing_high"])
        swing_low = None if pd.isna(row["recent_swing_low"]) else float(row["recent_swing_low"])
        swing_support_distance_atr = None if swing_low is None or not atr14 else abs(close - swing_low) / atr14
        swing_resistance_distance_atr = None if swing_high is None or not atr14 else abs(swing_high - close) / atr14
        if all(column in enriched.columns for column in POSITION_MAP_COLUMNS):
            band_upper = float(row["band_upper"])
            band_lower = float(row["band_lower"])
            band_volatility_unit = float(row["band_volatility_unit"])
            axis_distance_vol = float(row["axis_distance_vol"])
            ema55_distance_vol = float(row["ema55_distance_vol"])
            band_position = float(row["band_position"])
        else:
            band_upper, band_lower, band_volatility_unit = strategy._derive_position_bands(
                history_df,
                ema100=ema100,
                atr14=atr14,
            )
            axis_distance_vol = strategy._normalized_distance(close, ema100, band_volatility_unit)
            ema55_distance_vol = strategy._normalized_distance(close, ema55, band_volatility_unit)
            band_position = strategy._band_position(close, band_lower, band_upper)

        alignment = determine_ema_alignment(
            float(row["ema21"]),
            float(row["ema55"]),
            float(row["ema100"]),
            float(row["ema200"]),
        )
        bias = determine_trend_bias(close, float(row["ema200"]), alignment)

        notes: list[str] = []
        if bias == Bias.BULLISH:
            notes.append("价格站在 EMA200 上方，EMA21/55/100/200 结构仍偏多")
        elif bias == Bias.BEARISH:
            notes.append("价格压在 EMA200 下方，EMA21/55/100/200 结构仍偏空")
        else:
            notes.append("价格与 EMA21/55/100/200 没有形成清晰共振")
        notes.append(
            f"执行区使用 EMA21/EMA55：{round(float(row['execution_zone_low']), 4)} - {round(float(row['execution_zone_high']), 4)}"
        )
        notes.append(
            f"EMA100 中轴 band 为 {round(band_lower, 4)} - {round(band_upper, 4)}，"
            f"当前位置 band_position={round(band_position, 3)}"
        )

        model = TimeframeAnalysis(
            timeframe=timeframe,
            latest_timestamp=row["timestamp"].to_pydatetime(),
            close=close,
            ema21=float(row["ema21"]),
            ema55=float(row["ema55"]),
            ema100=float(row["ema100"]),
            ema200=float(row["ema200"]),
            atr14=atr14,
            atr_pct=round(float(row["atr_pct"]), 4),
            price_vs_ema21_pct=round(pct_distance(close, float(row["ema21"])), 4),
            price_vs_ema55_pct=round(pct_distance(close, float(row["ema55"])), 4),
            price_vs_ema100_pct=round(pct_distance(close, float(row["ema100"])), 4),
            price_vs_ema200_pct=round(pct_distance(close, float(row["ema200"])), 4),
            ema_alignment=alignment,
            trend_bias=bias,
            trend_score=int(row["trend_score"]),
            structure_state=StructureState(str(row["structure_state"])),
            swing_high=swing_high,
            swing_low=swing_low,
            is_pullback_to_value_area=bool(row["is_pullback_to_value_area"]),
            is_extended=bool(row["is_extended"]),
            notes=notes,
        )
        candle_profile = {
            "lookback": int(strategy.config["micro"]["confirmation_lookback"]),
            "latest_body_ratio": round(float(row["cp_body_ratio"]), 4),
            "latest_upper_wick_ratio": round(float(row["cp_upper_wick_ratio"]), 4),
            "latest_lower_wick_ratio": round(float(row["cp_lower_wick_ratio"]), 4),
            "latest_range_ratio": round(float(row["cp_range_ratio"]), 4),
            "volume_ratio": round(float(row["cp_volume_ratio"]), 4),
            "quote_volume_ratio": round(float(row["cp_quote_volume_ratio"]), 4),
            "is_volume_contracting": bool(row["cp_is_volume_contracting"]),
            "is_spiky": bool(row["cp_is_spiky"]),
            "is_doji": bool(row["cp_is_doji"]),
            "has_bullish_rejection": bool(row["cp_has_bullish_rejection"]),
            "has_bearish_rejection": bool(row["cp_has_bearish_rejection"]),
            "has_bullish_reversal_candle": bool(row["cp_has_bullish_reversal_candle"]),
            "has_bearish_reversal_candle": bool(row["cp_has_bearish_reversal_candle"]),
            "latest_quote_volume": round(float(row["cp_latest_quote_volume"]), 4),
            "median_quote_volume": round(float(row["cp_median_quote_volume"]), 4),
        }
        divergence_profile = divergence_profile_from_row(
            row,
            enabled=bool(strategy.config.get("divergence", {}).get("enabled", False)),
        )
        return PreparedTimeframe(
            model=model,
            debug={
                "position_map": {
                    "volatility_mode": str(strategy.config["position_map"]["band_volatility_mode"]),
                    "band_upper": round(band_upper, 4),
                    "band_lower": round(band_lower, 4),
                    "band_width": round(band_upper - band_lower, 4),
                    "band_volatility_unit": round(band_volatility_unit, 4),
                    "axis_distance_vol": round(axis_distance_vol, 4),
                    "ema55_distance_vol": round(ema55_distance_vol, 4),
                    "band_position": round(band_position, 4),
                }
            },
            df=history_df,
            value_zone_low=float(row["value_zone_low"]),
            value_zone_high=float(row["value_zone_high"]),
            execution_zone_low=float(row["execution_zone_low"]),
            execution_zone_high=float(row["execution_zone_high"]),
            distance_to_value_atr=float(row["distance_to_execution_atr"]),
            distance_to_execution_atr=float(row["distance_to_execution_atr"]),
            band_upper=band_upper,
            band_lower=band_lower,
            band_volatility_unit=band_volatility_unit,
            axis_distance_vol=axis_distance_vol,
            ema55_distance_vol=ema55_distance_vol,
            band_position=band_position,
            swing_support_distance_atr=swing_support_distance_atr,
            swing_resistance_distance_atr=swing_resistance_distance_atr,
            candle_profile=candle_profile,
            divergence_profile=divergence_profile,
        )

    def _open_pending_entry(
        self,
        *,
        symbol: str,
        strategy_profile: str,
        pending_entry: _PendingEntry,
        candle: pd.Series,
    ) -> Optional[_OpenPosition]:
        signal = pending_entry.signal
        if signal.stop_price is None or signal.tp1_price is None or signal.tp2_price is None:
            return None

        raw_open = float(candle["open"])
        if signal.action == Action.LONG:
            if raw_open <= signal.stop_price:
                return None
            fill_price = raw_open * (1 + self.assumptions.slippage_bps / 10000)
        else:
            if raw_open >= signal.stop_price:
                return None
            fill_price = raw_open * (1 - self.assumptions.slippage_bps / 10000)

        initial_risk = abs(fill_price - float(signal.stop_price))
        tp1_price = float(signal.tp1_price)
        tp2_price = float(signal.tp2_price)
        exit_config = self._resolve_exit_config(side=signal.action, strategy_profile=strategy_profile)
        take_profit_mode = str(exit_config.get("take_profit_mode", self.assumptions.take_profit_mode))
        fixed_take_profit_r = exit_config.get("fixed_take_profit_r")
        scaled_tp1_r = exit_config.get("scaled_tp1_r")
        scaled_tp2_r = exit_config.get("scaled_tp2_r")
        tp1_scale_out = float(exit_config.get("tp1_scale_out", self.assumptions.tp1_scale_out))
        move_stop_to_entry_after_tp1 = bool(
            exit_config.get("move_stop_to_entry_after_tp1", self.assumptions.move_stop_to_entry_after_tp1)
        )
        max_hold_bars = int(exit_config.get("max_hold_bars", self._default_max_hold_bars(strategy_profile)))

        if take_profit_mode == "fixed_r":
            target_r = float(fixed_take_profit_r or 0.0)
            if signal.action == Action.LONG:
                target_price = fill_price + (initial_risk * target_r)
            else:
                target_price = fill_price - (initial_risk * target_r)
            tp1_price = target_price
            tp2_price = target_price
        else:
            if scaled_tp1_r is not None:
                tp1_target_r = float(scaled_tp1_r)
                tp1_price = fill_price + (initial_risk * tp1_target_r) if signal.action == Action.LONG else fill_price - (initial_risk * tp1_target_r)
            if scaled_tp2_r is not None:
                tp2_target_r = float(scaled_tp2_r)
                tp2_price = fill_price + (initial_risk * tp2_target_r) if signal.action == Action.LONG else fill_price - (initial_risk * tp2_target_r)

        signal_bias = getattr(
            signal,
            "bias",
            Bias.BULLISH if signal.action == Action.LONG else Bias.BEARISH if signal.action == Action.SHORT else Bias.NEUTRAL,
        )
        signal_trend_strength = int(getattr(signal, "trend_strength", 0))
        position = _OpenPosition(
            symbol=symbol,
            strategy_profile=strategy_profile,
            side=signal.action,
            higher_bias=signal_bias,
            trend_strength=signal_trend_strength,
            signal_time=signal.timestamp,
            entry_time=candle["timestamp"].to_pydatetime(),
            entry_price=fill_price,
            initial_stop_price=float(signal.stop_price),
            current_stop_price=float(signal.stop_price),
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            take_profit_mode=take_profit_mode,
            fixed_take_profit_r=float(fixed_take_profit_r) if fixed_take_profit_r is not None else None,
            confidence=signal.confidence,
            tp1_scale_out=tp1_scale_out,
            move_stop_to_entry_after_tp1=move_stop_to_entry_after_tp1,
            max_hold_bars=max_hold_bars,
            trailing_stop_enabled=self.assumptions.trailing_stop_enabled,
            trailing_stop_atr_mult=self.assumptions.trailing_stop_atr_mult,
            trailing_stop_activation_r=self.assumptions.trailing_stop_activation_r,
            highest_price_since_entry=fill_price,
            lowest_price_since_entry=fill_price,
            leverage=self.assumptions.leverage,
        )
        entry_fee = fill_price * (self.assumptions.taker_fee_bps / 10000) * position.leverage
        position.fees_quote += entry_fee
        position.realized_pnl_quote -= entry_fee
        position.last_fill_price = fill_price
        return position

    def _update_open_position(
        self,
        *,
        position: _OpenPosition,
        candle: pd.Series,
        max_hold_bars: int,
    ) -> Optional[BacktestTrade]:
        max_hold_bars = int(position.max_hold_bars or max_hold_bars)
        position.bars_held += 1
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        # Track extreme prices for trailing stop
        position.highest_price_since_entry = max(position.highest_price_since_entry, high)
        position.lowest_price_since_entry = min(position.lowest_price_since_entry, low)

        # Update trailing stop if enabled and activated
        if position.trailing_stop_enabled:
            self._update_trailing_stop(position, candle)

        if position.take_profit_mode == "fixed_r":
            return self._update_open_position_fixed_r(position=position, candle=candle, max_hold_bars=max_hold_bars)

        if position.side == Action.LONG:
            if self.assumptions.conservative_same_bar_exit and low <= position.current_stop_price:
                reason = self._stop_exit_reason(position)
                return self._close_position(position, candle, exit_reason=reason, fill_price=position.current_stop_price)
            if (not position.tp1_hit) and high >= position.tp1_price:
                self._take_partial_profit(position, target_price=position.tp1_price, qty=position.tp1_scale_out)
                position.tp1_hit = True
                if not position.trailing_stop_active and position.move_stop_to_entry_after_tp1:
                    position.current_stop_price = max(position.current_stop_price, position.entry_price)
            if not position.trailing_stop_enabled and position.tp1_hit and high >= position.tp2_price:
                position.tp2_hit = True
                return self._close_position(position, candle, exit_reason="tp2", fill_price=position.tp2_price)
            if low <= position.current_stop_price:
                reason = self._stop_exit_reason(position)
                return self._close_position(position, candle, exit_reason=reason, fill_price=position.current_stop_price)
        else:
            if self.assumptions.conservative_same_bar_exit and high >= position.current_stop_price:
                reason = self._stop_exit_reason(position)
                return self._close_position(position, candle, exit_reason=reason, fill_price=position.current_stop_price)
            if (not position.tp1_hit) and low <= position.tp1_price:
                self._take_partial_profit(position, target_price=position.tp1_price, qty=position.tp1_scale_out)
                position.tp1_hit = True
                if not position.trailing_stop_active and position.move_stop_to_entry_after_tp1:
                    position.current_stop_price = min(position.current_stop_price, position.entry_price)
            if not position.trailing_stop_enabled and position.tp1_hit and low <= position.tp2_price:
                position.tp2_hit = True
                return self._close_position(position, candle, exit_reason="tp2", fill_price=position.tp2_price)
            if high >= position.current_stop_price:
                reason = self._stop_exit_reason(position)
                return self._close_position(position, candle, exit_reason=reason, fill_price=position.current_stop_price)

        if position.bars_held >= max_hold_bars:
            return self._close_position(position, candle, exit_reason="time_stop", fill_price=close)
        return None

    def _stop_exit_reason(self, position: _OpenPosition) -> str:
        if position.trailing_stop_active:
            return "trailing_stop"
        if position.tp1_hit:
            if position.side == Action.LONG and position.current_stop_price >= position.entry_price:
                return "breakeven_after_tp1"
            if position.side == Action.SHORT and position.current_stop_price <= position.entry_price:
                return "breakeven_after_tp1"
        return "stop_loss"

    def _update_trailing_stop(self, position: _OpenPosition, candle: pd.Series) -> None:
        """Update trailing stop based on ATR and highest/lowest price since entry."""
        initial_risk = abs(position.entry_price - position.initial_stop_price)
        if initial_risk <= 0:
            return

        # Get ATR from candle if available, otherwise use initial risk as proxy
        atr = float(candle.get("atr14", initial_risk))

        if position.side == Action.LONG:
            # Check if profit has reached activation threshold
            unrealized_r = (position.highest_price_since_entry - position.entry_price) / initial_risk
            if unrealized_r >= position.trailing_stop_activation_r:
                position.trailing_stop_active = True
                trailing_level = position.highest_price_since_entry - (atr * position.trailing_stop_atr_mult)
                position.current_stop_price = max(position.current_stop_price, trailing_level)
        else:
            unrealized_r = (position.entry_price - position.lowest_price_since_entry) / initial_risk
            if unrealized_r >= position.trailing_stop_activation_r:
                position.trailing_stop_active = True
                trailing_level = position.lowest_price_since_entry + (atr * position.trailing_stop_atr_mult)
                position.current_stop_price = min(position.current_stop_price, trailing_level)

    def _update_open_position_fixed_r(
        self,
        *,
        position: _OpenPosition,
        candle: pd.Series,
        max_hold_bars: int,
    ) -> Optional[BacktestTrade]:
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        exit_reason = f"take_profit_{float(position.fixed_take_profit_r or 0.0):g}R"

        if position.side == Action.LONG:
            if self.assumptions.conservative_same_bar_exit and low <= position.current_stop_price:
                return self._close_position(position, candle, exit_reason="stop_loss", fill_price=position.current_stop_price)
            if high >= position.tp1_price:
                position.tp1_hit = True
                position.tp2_hit = True
                return self._close_position(position, candle, exit_reason=exit_reason, fill_price=position.tp1_price)
            if low <= position.current_stop_price:
                return self._close_position(position, candle, exit_reason="stop_loss", fill_price=position.current_stop_price)
        else:
            if self.assumptions.conservative_same_bar_exit and high >= position.current_stop_price:
                return self._close_position(position, candle, exit_reason="stop_loss", fill_price=position.current_stop_price)
            if low <= position.tp1_price:
                position.tp1_hit = True
                position.tp2_hit = True
                return self._close_position(position, candle, exit_reason=exit_reason, fill_price=position.tp1_price)
            if high >= position.current_stop_price:
                return self._close_position(position, candle, exit_reason="stop_loss", fill_price=position.current_stop_price)

        if position.bars_held >= max_hold_bars:
            return self._close_position(position, candle, exit_reason="time_stop", fill_price=close)
        return None

    def _take_partial_profit(self, position: _OpenPosition, *, target_price: float, qty: float) -> None:
        if qty <= 0 or position.remaining_qty <= 0:
            return
        fill_qty = min(position.remaining_qty, qty)
        effective_price = self._effective_exit_price(position.side, target_price)
        gross = self._gross_pnl(position.side, position.entry_price, effective_price, fill_qty)
        gross *= position.leverage
        fee = effective_price * fill_qty * (self.assumptions.taker_fee_bps / 10000) * position.leverage
        position.realized_pnl_quote += gross - fee
        position.fees_quote += fee
        position.remaining_qty -= fill_qty
        position.last_fill_price = effective_price

    def _close_position(
        self,
        position: _OpenPosition,
        candle: pd.Series,
        *,
        exit_reason: str,
        fill_price: float,
    ) -> BacktestTrade:
        effective_price = self._effective_exit_price(position.side, fill_price)
        if position.remaining_qty > 0:
            gross = self._gross_pnl(position.side, position.entry_price, effective_price, position.remaining_qty)
            gross *= position.leverage
            fee = effective_price * position.remaining_qty * (self.assumptions.taker_fee_bps / 10000) * position.leverage
            position.realized_pnl_quote += gross - fee
            position.fees_quote += fee
            position.remaining_qty = 0.0
        position.last_fill_price = effective_price

        initial_risk = abs(position.entry_price - position.initial_stop_price)
        pnl_r = position.realized_pnl_quote / initial_risk if initial_risk else 0.0
        pnl_pct = (position.realized_pnl_quote / position.entry_price) * 100 if position.entry_price else 0.0
        return BacktestTrade(
            symbol=position.symbol,
            strategy_profile=position.strategy_profile,
            side=position.side.value,
            higher_bias=position.higher_bias.value,
            trend_strength=position.trend_strength,
            signal_time=position.signal_time.isoformat(),
            entry_time=position.entry_time.isoformat(),
            exit_time=candle["timestamp"].to_pydatetime().isoformat(),
            entry_price=round(position.entry_price, 6),
            exit_price=round(effective_price, 6),
            stop_price=round(position.initial_stop_price, 6),
            tp1_price=round(position.tp1_price, 6),
            tp2_price=round(position.tp2_price, 6),
            bars_held=position.bars_held,
            exit_reason=exit_reason,
            confidence=position.confidence,
            tp1_hit=position.tp1_hit,
            tp2_hit=position.tp2_hit,
            pnl_pct=round(pnl_pct, 4),
            pnl_r=round(pnl_r, 4),
            gross_pnl_quote=round(position.realized_pnl_quote, 6),
            fees_quote=round(position.fees_quote, 6),
            leverage=position.leverage,
        )

    def _effective_exit_price(self, side: Action, raw_price: float) -> float:
        if side == Action.LONG:
            return raw_price * (1 - self.assumptions.slippage_bps / 10000)
        return raw_price * (1 + self.assumptions.slippage_bps / 10000)

    @staticmethod
    def _gross_pnl(side: Action, entry_price: float, exit_price: float, qty: float) -> float:
        if side == Action.LONG:
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty

    def _max_hold_bars(self, strategy_profile: str) -> int:
        return self._default_max_hold_bars(strategy_profile)

    def _summarize_trades(
        self,
        *,
        trades: list[BacktestTrade],
        strategy_profile: str,
        symbol: Optional[str],
        signals_now: int,
        skipped_entries: int,
    ) -> BacktestSummary:
        if not trades:
            return BacktestSummary(
                strategy_profile=strategy_profile,
                symbol=symbol,
                total_trades=0,
                wins=0,
                losses=0,
                breakeven=0,
                win_rate=0.0,
                payoff_ratio=0.0,
                profit_factor=0.0,
                expectancy_r=0.0,
                avg_r=0.0,
                median_r=0.0,
                cumulative_r=0.0,
                cumulative_return_pct=0.0,
                max_drawdown_r=0.0,
                avg_holding_bars=0.0,
                avg_holding_hours=0.0,
                tp1_hit_rate=0.0,
                tp2_hit_rate=0.0,
                signals_now=signals_now,
                skipped_entries=skipped_entries,
            )

        pnl_rs = [item.pnl_r for item in trades]
        winners = [item.pnl_r for item in trades if item.pnl_r > 0]
        losers = [item.pnl_r for item in trades if item.pnl_r < 0]
        wins = len(winners)
        losses = len(losers)
        breakeven = len(trades) - wins - losses
        cumulative_r = float(sum(pnl_rs))
        cumulative_return_pct = float(sum(item.pnl_pct for item in trades))
        profit_factor = (sum(winners) / abs(sum(losers))) if losers else 0.0
        payoff_ratio = (mean(winners) / abs(mean(losers))) if winners and losers else 0.0
        equity_curve = np.cumsum(pnl_rs)
        running_peak = np.maximum.accumulate(equity_curve)
        drawdowns = running_peak - equity_curve
        avg_bars = mean([item.bars_held for item in trades])
        hours_per_bar = 3 / 60 if strategy_profile == "intraday_mtf_v1" else 1

        return BacktestSummary(
            strategy_profile=strategy_profile,
            symbol=symbol,
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            breakeven=breakeven,
            win_rate=round((wins / len(trades)) * 100, 2),
            payoff_ratio=round(payoff_ratio, 4),
            profit_factor=round(profit_factor, 4),
            expectancy_r=round(mean(pnl_rs), 4),
            avg_r=round(mean(pnl_rs), 4),
            median_r=round(median(pnl_rs), 4),
            cumulative_r=round(cumulative_r, 4),
            cumulative_return_pct=round(cumulative_return_pct, 4),
            max_drawdown_r=round(float(drawdowns.max()) if len(drawdowns) else 0.0, 4),
            avg_holding_bars=round(avg_bars, 2),
            avg_holding_hours=round(avg_bars * hours_per_bar, 2),
            tp1_hit_rate=round((sum(1 for item in trades if item.tp1_hit) / len(trades)) * 100, 2),
            tp2_hit_rate=round((sum(1 for item in trades if item.tp2_hit) / len(trades)) * 100, 2),
            signals_now=signals_now,
            skipped_entries=skipped_entries,
        )

    def _cache_path(self, *, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
        safe_symbol = symbol.lower().replace("/", "_").replace(":", "_")
        filename = f"{safe_symbol}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        return self.cache_dir / filename

    @staticmethod
    def save_report(report: BacktestReport, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_profile = str(report.assumptions.get("exit_profile", "default"))
        safe_profile = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw_profile).strip("_") or "default"
        json_path = output_dir / f"backtest_{safe_profile}_{stamp}.json"
        csv_path = output_dir / f"backtest_{safe_profile}_{stamp}_trades.csv"
        json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        trades_df = pd.DataFrame([asdict(item) for item in report.trades])
        trades_df.to_csv(csv_path, index=False)
        return json_path, csv_path
