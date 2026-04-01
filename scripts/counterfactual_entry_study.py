from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Action, Bias, TriggerState, VolatilityState
from app.services.strategy_service import StrategyService
from app.strategies.scoring import ScoreCard
from app.strategies.windowed_mtf import WindowedMTFStrategy
from scripts.event_study_setup import bucket_trend_strength


DEFAULT_HORIZON = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare counterfactual entry styles on the same setup-ready episodes."
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--strategy-profile", default="swing_trend_v1")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default=None, help="UTC start date, e.g. 2024-03-19")
    parser.add_argument("--end", default=None, help="UTC end date, e.g. 2026-03-19")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--horizon-bars", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--output-dir",
        default="artifacts/event_studies/counterfactual_entries",
        help="Directory for report outputs.",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _setup_ready(setup_assessment: dict[str, Any], *, higher_bias: Bias, trend_friendly: bool) -> bool:
    if higher_bias == Bias.NEUTRAL or not trend_friendly:
        return False
    if not bool(setup_assessment.get("aligned")):
        return False
    if not bool(setup_assessment.get("pullback_ready")):
        return False
    if bool(setup_assessment.get("require_reversal_candle")) and not bool(setup_assessment.get("reversal_ready", True)):
        return False
    if bool(setup_assessment.get("require_divergence_gate")) and not bool(setup_assessment.get("divergence_ready", True)):
        return False
    return True


def _confidence(
    *,
    strategy: WindowedMTFStrategy,
    higher_bias: Bias,
    setup_assessment: dict[str, Any],
    trigger_assessment: dict[str, Any],
    volatility_state: VolatilityState,
) -> int:
    setup_key = str(strategy.window_config["setup_timeframe"])
    trigger_key = str(strategy.window_config["trigger_timeframe"])
    higher_label = strategy._format_timeframe_group(tuple(strategy.window_config["higher_timeframes"]))
    scorecard = ScoreCard(base=50)
    if higher_bias == Bias.BULLISH:
        scorecard.add(15, "higher_bias", f"{higher_label} 同步偏多")
    elif higher_bias == Bias.BEARISH:
        scorecard.add(15, "higher_bias", f"{higher_label} 同步偏空")
    else:
        scorecard.add(-20, "higher_conflict", f"{higher_label} 没有形成同向共振")
    scorecard.add(setup_assessment["score"], f"{setup_key}_setup", setup_assessment["score_note"])
    scorecard.add(trigger_assessment["score"], f"{trigger_key}_trigger", trigger_assessment["score_note"])
    if volatility_state == VolatilityState.HIGH:
        scorecard.add(-15, "volatility", f"{setup_key} ATR 百分比偏高")
    elif volatility_state == VolatilityState.LOW:
        scorecard.add(3, "volatility", f"{setup_key} 波动可控")
    return int(scorecard.total)


def build_setup_timeline(
    *,
    service: BacktestService,
    exchange: str,
    market_type: str,
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    strategy = service.strategy_service.build_strategy(strategy_profile)
    if not isinstance(strategy, WindowedMTFStrategy):
        raise TypeError(f"Counterfactual entry study supports WindowedMTFStrategy only, got {strategy_profile}")

    frames = service._load_history(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        strategy_profile=strategy_profile,
        start=start,
        end=end,
    )
    enriched = {
        timeframe: service._enrich_frame(strategy, timeframe, frame)
        for timeframe, frame in frames.items()
    }

    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    setup_key = str(strategy.window_config["setup_timeframe"])
    reference_key = str(strategy.window_config.get("reference_timeframe", setup_key))
    required = tuple(strategy.required_timeframes)
    trigger_frame = enriched[trigger_tf]
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}
    min_required = max(int(service.assumptions.lookback // 3), 20)
    action_threshold = int(strategy.config["confidence"]["action_threshold"])

    rows: list[dict[str, Any]] = []
    for trigger_idx in range(len(trigger_frame)):
        candle = trigger_frame.iloc[trigger_idx]
        ts = candle["timestamp"].to_pydatetime()
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
        if not ready or any(index < min_required for index in current_indices.values()):
            continue

        prepared = {
            timeframe: service._build_snapshot(strategy, timeframe, enriched[timeframe], current_indices[timeframe])
            for timeframe in current_indices
        }
        higher_bias, trend_strength = strategy._derive_higher_timeframe_bias(prepared)
        volatility_state = strategy._derive_volatility_state(prepared[setup_key])
        trend_friendly = strategy._is_trend_friendly(
            higher_bias=higher_bias,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
        )
        setup_assessment = strategy._assess_setup(higher_bias, prepared[setup_key], setup_key)
        trigger_assessment = strategy._assess_trigger(higher_bias, prepared[trigger_tf], trigger_tf)
        confidence = _confidence(
            strategy=strategy,
            higher_bias=higher_bias,
            setup_assessment=setup_assessment,
            trigger_assessment=trigger_assessment,
            volatility_state=volatility_state,
        )
        setup_ready = _setup_ready(setup_assessment, higher_bias=higher_bias, trend_friendly=trend_friendly)
        trigger_confirmed = setup_ready and trigger_assessment["state"] in {
            TriggerState.BULLISH_CONFIRMED,
            TriggerState.BEARISH_CONFIRMED,
        }
        confidence_pass = confidence >= action_threshold

        side: Optional[str]
        if higher_bias == Bias.BULLISH:
            side = "LONG"
        elif higher_bias == Bias.BEARISH:
            side = "SHORT"
        else:
            side = None

        setup_stop_price = None
        if setup_ready and side is not None:
            forced_action = Action.LONG if side == "LONG" else Action.SHORT
            trade_plan = strategy._build_trade_plan(
                action=forced_action,
                bias=higher_bias,
                setup_ctx=prepared[setup_key],
                reference_ctx=prepared[reference_key],
                current_price=prepared[trigger_tf].model.close,
                setup_key=setup_key,
                reference_key=reference_key,
            )
            if trade_plan["stop_loss"] is not None:
                setup_stop_price = float(trade_plan["stop_loss"].price)

        rows.append(
            {
                "timestamp": pd.Timestamp(ts),
                "trigger_index": trigger_idx,
                "side": side,
                "higher_bias": higher_bias.value,
                "trend_strength": int(trend_strength),
                "trend_strength_bucket": bucket_trend_strength(int(trend_strength)),
                "volatility_state": volatility_state.value,
                "trend_friendly": bool(trend_friendly),
                "setup_ready": bool(setup_ready),
                "trigger_confirmed": bool(trigger_confirmed and confidence_pass),
                "confidence": int(confidence),
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "execution_zone_low": float(prepared[setup_key].execution_zone_low),
                "execution_zone_high": float(prepared[setup_key].execution_zone_high),
                "setup_stop_price": setup_stop_price,
                "trigger_high": float(candle["high"]),
                "trigger_low": float(candle["low"]),
            }
        )

    timeline = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return timeline


def extract_setup_episodes(timeline: pd.DataFrame) -> list[dict[str, Any]]:
    if timeline.empty:
        return []

    episodes: list[dict[str, Any]] = []
    active_start: Optional[int] = None
    active_side: Optional[str] = None

    for idx, row in timeline.iterrows():
        is_ready = bool(row["setup_ready"]) and row["side"] is not None and row["setup_stop_price"] is not None
        side = str(row["side"]) if row["side"] is not None else None
        if is_ready and active_start is None:
            active_start = idx
            active_side = side
            continue
        if is_ready and active_start is not None and side == active_side:
            continue
        if active_start is not None:
            episodes.append(
                {
                    "episode_id": len(episodes),
                    "start_idx": active_start,
                    "end_idx": idx - 1,
                }
            )
            active_start = None
            active_side = None
        if is_ready:
            active_start = idx
            active_side = side

    if active_start is not None:
        episodes.append(
            {
                "episode_id": len(episodes),
                "start_idx": active_start,
                "end_idx": int(timeline.index[-1]),
            }
        )
    return episodes


def _directional_price_advantage(side: str, reference_price: float, candidate_price: float, baseline_risk: float) -> Optional[float]:
    if baseline_risk <= 0:
        return None
    if side == "LONG":
        return round((candidate_price - reference_price) / baseline_risk, 4)
    return round((reference_price - candidate_price) / baseline_risk, 4)


def _market_fill_price(side: str, raw_open: float, slippage_bps: float) -> float:
    if side == "LONG":
        return raw_open * (1 + slippage_bps / 10000)
    return raw_open * (1 - slippage_bps / 10000)


def _simulate_market_next_open(
    *,
    timeline: pd.DataFrame,
    signal_idx: int,
    side: str,
    stop_price: float,
    slippage_bps: float,
) -> dict[str, Any]:
    fill_idx = signal_idx + 1
    if fill_idx >= len(timeline):
        return {"filled": False, "fill_reason": "no_next_bar"}

    bar = timeline.iloc[fill_idx]
    raw_open = float(bar["open"])
    if side == "LONG" and raw_open <= stop_price:
        return {"filled": False, "fill_reason": "gapped_through_stop"}
    if side == "SHORT" and raw_open >= stop_price:
        return {"filled": False, "fill_reason": "gapped_through_stop"}

    entry_price = _market_fill_price(side, raw_open, slippage_bps)
    return {
        "filled": True,
        "fill_reason": "filled_next_open",
        "fill_idx": fill_idx,
        "fill_timestamp": timeline.iloc[fill_idx]["timestamp"].isoformat(),
        "entry_price": round(entry_price, 6),
        "intrabar_fill": False,
        "evaluate_from_same_bar": True,
    }


def _simulate_setup_close_reference(
    *,
    timeline: pd.DataFrame,
    signal_idx: int,
    side: str,
    stop_price: float,
) -> dict[str, Any]:
    bar = timeline.iloc[signal_idx]
    close_price = float(bar["close"])
    if side == "LONG" and close_price <= stop_price:
        return {"filled": False, "fill_reason": "close_below_stop"}
    if side == "SHORT" and close_price >= stop_price:
        return {"filled": False, "fill_reason": "close_above_stop"}
    return {
        "filled": True,
        "fill_reason": "filled_setup_close_reference",
        "fill_idx": signal_idx,
        "fill_timestamp": timeline.iloc[signal_idx]["timestamp"].isoformat(),
        "entry_price": round(close_price, 6),
        "intrabar_fill": False,
        "evaluate_from_same_bar": False,
    }


def _simulate_zone_limit(
    *,
    timeline: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    side: str,
    stop_price: float,
    limit_price: float,
) -> dict[str, Any]:
    for fill_idx in range(start_idx + 1, end_idx + 1):
        bar = timeline.iloc[fill_idx]
        raw_open = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])

        if side == "LONG":
            if raw_open <= stop_price:
                return {"filled": False, "fill_reason": "invalidated_before_limit"}
            if raw_open <= limit_price:
                return {
                    "filled": True,
                    "fill_reason": "gap_fill_better_than_limit",
                    "fill_idx": fill_idx,
                    "fill_timestamp": bar["timestamp"].isoformat(),
                    "entry_price": round(raw_open, 6),
                    "intrabar_fill": False,
                    "evaluate_from_same_bar": True,
                }
            if low <= limit_price:
                same_bar_stop = low <= stop_price
                return {
                    "filled": True,
                    "fill_reason": "limit_touched",
                    "fill_idx": fill_idx,
                    "fill_timestamp": bar["timestamp"].isoformat(),
                    "entry_price": round(limit_price, 6),
                    "intrabar_fill": True,
                    "evaluate_from_same_bar": False,
                    "same_bar_stop": same_bar_stop,
                }
        else:
            if raw_open >= stop_price:
                return {"filled": False, "fill_reason": "invalidated_before_limit"}
            if raw_open >= limit_price:
                return {
                    "filled": True,
                    "fill_reason": "gap_fill_better_than_limit",
                    "fill_idx": fill_idx,
                    "fill_timestamp": bar["timestamp"].isoformat(),
                    "entry_price": round(raw_open, 6),
                    "intrabar_fill": False,
                    "evaluate_from_same_bar": True,
                }
            if high >= limit_price:
                same_bar_stop = high >= stop_price
                return {
                    "filled": True,
                    "fill_reason": "limit_touched",
                    "fill_idx": fill_idx,
                    "fill_timestamp": bar["timestamp"].isoformat(),
                    "entry_price": round(limit_price, 6),
                    "intrabar_fill": True,
                    "evaluate_from_same_bar": False,
                    "same_bar_stop": same_bar_stop,
                }
    return {"filled": False, "fill_reason": "limit_never_filled"}


def _simulate_trigger_breakout(
    *,
    timeline: pd.DataFrame,
    trigger_idx: int,
    end_idx: int,
    side: str,
    stop_price: float,
    breakout_level: float,
) -> dict[str, Any]:
    for fill_idx in range(trigger_idx + 1, end_idx + 1):
        bar = timeline.iloc[fill_idx]
        raw_open = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])

        if side == "LONG":
            if raw_open <= stop_price:
                return {"filled": False, "fill_reason": "invalidated_before_breakout"}
            if raw_open >= breakout_level:
                return {
                    "filled": True,
                    "fill_reason": "gap_breakout_fill",
                    "fill_idx": fill_idx,
                    "fill_timestamp": bar["timestamp"].isoformat(),
                    "entry_price": round(raw_open, 6),
                    "intrabar_fill": False,
                    "evaluate_from_same_bar": True,
                }
            if low <= stop_price:
                return {"filled": False, "fill_reason": "invalidated_before_breakout"}
            if high >= breakout_level:
                return {
                    "filled": True,
                    "fill_reason": "breakout_touched",
                    "fill_idx": fill_idx,
                    "fill_timestamp": bar["timestamp"].isoformat(),
                    "entry_price": round(breakout_level, 6),
                    "intrabar_fill": True,
                    "evaluate_from_same_bar": False,
                }
        else:
            if raw_open >= stop_price:
                return {"filled": False, "fill_reason": "invalidated_before_breakout"}
            if raw_open <= breakout_level:
                return {
                    "filled": True,
                    "fill_reason": "gap_breakout_fill",
                    "fill_idx": fill_idx,
                    "fill_timestamp": bar["timestamp"].isoformat(),
                    "entry_price": round(raw_open, 6),
                    "intrabar_fill": False,
                    "evaluate_from_same_bar": True,
                }
            if high >= stop_price:
                return {"filled": False, "fill_reason": "invalidated_before_breakout"}
            if low <= breakout_level:
                return {
                    "filled": True,
                    "fill_reason": "breakout_touched",
                    "fill_idx": fill_idx,
                    "fill_timestamp": bar["timestamp"].isoformat(),
                    "entry_price": round(breakout_level, 6),
                    "intrabar_fill": True,
                    "evaluate_from_same_bar": False,
                }
    return {"filled": False, "fill_reason": "breakout_never_triggered"}


def _path_metrics(
    *,
    timeline: pd.DataFrame,
    fill_idx: int,
    side: str,
    entry_price: float,
    stop_price: float,
    horizon_bars: int,
    evaluate_from_same_bar: bool,
    same_bar_stop: bool,
) -> dict[str, Any]:
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return {
            "risk_abs": 0.0,
            "hit_plus_1r_before_stop": False,
            "hit_plus_2r_before_stop": False,
            "plus_1r_first": False,
            "plus_2r_first": False,
            "stop_hit_first": False,
            "mfe_r": None,
            "mae_r": None,
        }

    if side == "LONG":
        plus_1r = entry_price + risk
        plus_2r = entry_price + 2 * risk
    else:
        plus_1r = entry_price - risk
        plus_2r = entry_price - 2 * risk

    if same_bar_stop:
        return {
            "risk_abs": round(risk, 6),
            "hit_plus_1r_before_stop": False,
            "hit_plus_2r_before_stop": False,
            "plus_1r_first": False,
            "plus_2r_first": False,
            "stop_hit_first": True,
            "mfe_r": 0.0,
            "mae_r": -1.0,
        }

    start_idx = fill_idx if evaluate_from_same_bar else fill_idx + 1
    end_idx = min(len(timeline) - 1, fill_idx + horizon_bars)
    if start_idx > end_idx:
        return {
            "risk_abs": round(risk, 6),
            "hit_plus_1r_before_stop": False,
            "hit_plus_2r_before_stop": False,
            "plus_1r_first": False,
            "plus_2r_first": False,
            "stop_hit_first": False,
            "mfe_r": None,
            "mae_r": None,
        }

    window = timeline.iloc[start_idx : end_idx + 1]
    if side == "LONG":
        mfe_r = (float(window["high"].max()) - entry_price) / risk
        mae_r = (float(window["low"].min()) - entry_price) / risk
    else:
        mfe_r = (entry_price - float(window["low"].min())) / risk
        mae_r = (entry_price - float(window["high"].max())) / risk

    hit_plus_1r_before_stop = False
    hit_plus_2r_before_stop = False
    stop_hit_first = False
    plus_1r_first = False
    plus_2r_first = False

    for _, bar in window.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        if side == "LONG":
            if low <= stop_price:
                stop_hit_first = True
                break
            if high >= plus_2r:
                hit_plus_1r_before_stop = True
                hit_plus_2r_before_stop = True
                plus_1r_first = True
                plus_2r_first = True
                break
            if high >= plus_1r:
                hit_plus_1r_before_stop = True
                plus_1r_first = True
        else:
            if high >= stop_price:
                stop_hit_first = True
                break
            if low <= plus_2r:
                hit_plus_1r_before_stop = True
                hit_plus_2r_before_stop = True
                plus_1r_first = True
                plus_2r_first = True
                break
            if low <= plus_1r:
                hit_plus_1r_before_stop = True
                plus_1r_first = True

    return {
        "risk_abs": round(risk, 6),
        "hit_plus_1r_before_stop": bool(hit_plus_1r_before_stop),
        "hit_plus_2r_before_stop": bool(hit_plus_2r_before_stop),
        "plus_1r_first": bool(plus_1r_first),
        "plus_2r_first": bool(plus_2r_first),
        "stop_hit_first": bool(stop_hit_first),
        "mfe_r": round(float(mfe_r), 4),
        "mae_r": round(float(mae_r), 4),
    }


def build_counterfactual_records(
    *,
    timeline: pd.DataFrame,
    episodes: list[dict[str, Any]],
    symbol: str,
    slippage_bps: float,
    horizon_bars: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for episode in episodes:
        start_idx = int(episode["start_idx"])
        end_idx = int(episode["end_idx"])
        setup_row = timeline.iloc[start_idx]
        side = str(setup_row["side"])
        stop_price = float(setup_row["setup_stop_price"])
        setup_close = _simulate_setup_close_reference(
            timeline=timeline,
            signal_idx=start_idx,
            side=side,
            stop_price=stop_price,
        )
        setup_market = _simulate_market_next_open(
            timeline=timeline,
            signal_idx=start_idx,
            side=side,
            stop_price=stop_price,
            slippage_bps=slippage_bps,
        )
        if side == "LONG":
            limit_price = float(setup_row["execution_zone_high"])
        else:
            limit_price = float(setup_row["execution_zone_low"])
        setup_limit = _simulate_zone_limit(
            timeline=timeline,
            start_idx=start_idx,
            end_idx=end_idx,
            side=side,
            stop_price=stop_price,
            limit_price=limit_price,
        )

        trigger_candidates = timeline.iloc[start_idx : end_idx + 1]
        trigger_rows = trigger_candidates.loc[trigger_candidates["trigger_confirmed"]]
        trigger_market: dict[str, Any]
        trigger_breakout: dict[str, Any]
        trigger_idx: Optional[int] = None
        breakout_level: Optional[float] = None
        if trigger_rows.empty:
            trigger_market = {"filled": False, "fill_reason": "trigger_never_confirmed"}
            trigger_breakout = {"filled": False, "fill_reason": "trigger_never_confirmed"}
        else:
            trigger_idx = int(trigger_rows.index[0])
            trigger_market = _simulate_market_next_open(
                timeline=timeline,
                signal_idx=trigger_idx,
                side=side,
                stop_price=stop_price,
                slippage_bps=slippage_bps,
            )
            trigger_bar = timeline.iloc[trigger_idx]
            breakout_level = float(trigger_bar["high"] if side == "LONG" else trigger_bar["low"])
            trigger_breakout = _simulate_trigger_breakout(
                timeline=timeline,
                trigger_idx=trigger_idx,
                end_idx=end_idx,
                side=side,
                stop_price=stop_price,
                breakout_level=breakout_level,
            )

        methods = {
            "setup_close_reference": setup_close,
            "setup_next_open_market": setup_market,
            "setup_execution_zone_limit": setup_limit,
            "trigger_confirm_next_open": trigger_market,
            "trigger_breakout": trigger_breakout,
        }

        baseline_entry = setup_market.get("entry_price") if setup_market.get("filled") else None
        baseline_risk = None
        if setup_market.get("filled"):
            baseline_risk = abs(float(setup_market["entry_price"]) - stop_price)

        for method, result in methods.items():
            record = {
                "episode_id": int(episode["episode_id"]),
                "method": method,
                "symbol": symbol,
                "side": side,
                "setup_timestamp": setup_row["timestamp"].isoformat(),
                "trend_strength": int(setup_row["trend_strength"]),
                "trend_strength_bucket": str(setup_row["trend_strength_bucket"]),
                "setup_stop_price": round(stop_price, 6),
                "setup_execution_zone_low": round(float(setup_row["execution_zone_low"]), 6),
                "setup_execution_zone_high": round(float(setup_row["execution_zone_high"]), 6),
                "trigger_timestamp": timeline.iloc[trigger_idx]["timestamp"].isoformat() if trigger_idx is not None else None,
                "trigger_breakout_level": round(breakout_level, 6) if breakout_level is not None else None,
                "filled": bool(result.get("filled", False)),
                "fill_reason": result.get("fill_reason"),
                "fill_timestamp": result.get("fill_timestamp"),
                "fill_bars_after_setup": (int(result["fill_idx"]) - start_idx) if result.get("filled") else None,
            }
            if result.get("filled"):
                entry_price = float(result["entry_price"])
                path = _path_metrics(
                    timeline=timeline,
                    fill_idx=int(result["fill_idx"]),
                    side=side,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    horizon_bars=horizon_bars,
                    evaluate_from_same_bar=bool(result.get("evaluate_from_same_bar", False)),
                    same_bar_stop=bool(result.get("same_bar_stop", False)),
                )
                record.update(
                    {
                        "entry_price": round(entry_price, 6),
                        "risk_abs": path["risk_abs"],
                        "hit_plus_1r_before_stop": path["hit_plus_1r_before_stop"],
                        "hit_plus_2r_before_stop": path["hit_plus_2r_before_stop"],
                        "plus_1r_first": path["plus_1r_first"],
                        "plus_2r_first": path["plus_2r_first"],
                        "stop_hit_first": path["stop_hit_first"],
                        "mfe_r": path["mfe_r"],
                        "mae_r": path["mae_r"],
                        "entry_worse_r_vs_setup_market": (
                            _directional_price_advantage(side, float(baseline_entry), entry_price, float(baseline_risk))
                            if baseline_entry is not None and baseline_risk is not None
                            else None
                        ),
                        "stop_distance_change_pct_vs_setup_market": (
                            round(((path["risk_abs"] / float(baseline_risk)) - 1) * 100, 2)
                            if baseline_risk not in {None, 0}
                            else None
                        ),
                    }
                )
            else:
                record.update(
                    {
                        "entry_price": None,
                        "risk_abs": None,
                        "hit_plus_1r_before_stop": None,
                        "hit_plus_2r_before_stop": None,
                        "plus_1r_first": None,
                        "plus_2r_first": None,
                        "stop_hit_first": None,
                        "mfe_r": None,
                        "mae_r": None,
                        "entry_worse_r_vs_setup_market": None,
                        "stop_distance_change_pct_vs_setup_market": None,
                    }
                )
            records.append(record)

    return pd.DataFrame(records)


def summarize_records(records: pd.DataFrame, group_by: list[str]) -> list[dict[str, Any]]:
    if records.empty:
        return []
    summaries: list[dict[str, Any]] = []
    for keys, group in records.groupby(group_by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {column: value for column, value in zip(group_by, keys)}
        total = int(len(group))
        filled = group.loc[group["filled"]]
        item["episodes"] = total
        item["filled"] = int(filled["filled"].sum())
        item["fill_rate_pct"] = round((item["filled"] / total) * 100, 2) if total else None
        item["avg_fill_bars"] = round(float(filled["fill_bars_after_setup"].mean()), 2) if not filled.empty else None
        item["avg_entry_worse_r_vs_setup_market"] = (
            round(float(filled["entry_worse_r_vs_setup_market"].dropna().mean()), 4)
            if not filled.empty and not filled["entry_worse_r_vs_setup_market"].dropna().empty
            else None
        )
        item["avg_stop_distance_change_pct_vs_setup_market"] = (
            round(float(filled["stop_distance_change_pct_vs_setup_market"].dropna().mean()), 2)
            if not filled.empty and not filled["stop_distance_change_pct_vs_setup_market"].dropna().empty
            else None
        )
        item["plus_1r_hit_rate_pct"] = (
            round(float(filled["hit_plus_1r_before_stop"].mean() * 100), 2)
            if not filled.empty and not filled["hit_plus_1r_before_stop"].dropna().empty
            else None
        )
        item["plus_2r_hit_rate_pct"] = (
            round(float(filled["hit_plus_2r_before_stop"].mean() * 100), 2)
            if not filled.empty and not filled["hit_plus_2r_before_stop"].dropna().empty
            else None
        )
        item["stop_first_rate_pct"] = (
            round(float(filled["stop_hit_first"].mean() * 100), 2)
            if not filled.empty and not filled["stop_hit_first"].dropna().empty
            else None
        )
        item["mean_mfe_r"] = round(float(filled["mfe_r"].dropna().mean()), 4) if not filled.empty and not filled["mfe_r"].dropna().empty else None
        item["mean_mae_r"] = round(float(filled["mae_r"].dropna().mean()), 4) if not filled.empty and not filled["mae_r"].dropna().empty else None
        summaries.append(item)
    return summaries


def render_markdown(
    *,
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
    horizon_bars: int,
    episodes: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    by_side: list[dict[str, Any]],
    by_bucket: list[dict[str, Any]],
) -> str:
    lines = [
        "# Counterfactual Entry Study",
        "",
        f"- 标的: {symbol}",
        f"- 策略: {strategy_profile}",
        f"- 回测窗口: {start.isoformat()} -> {end.isoformat()}",
        f"- Setup episodes: {len(episodes)}",
        f"- Horizon: {horizon_bars} bars",
        "",
        "## 方法说明",
        "",
        "- `setup_close_reference`: setup 当根收盘价参考入场，仅作为理论上界，不代表可执行回测。",
        "- `setup_next_open_market`: setup 下一根开盘直接市价进，这是主对照基线。",
        "- `setup_execution_zone_limit`: setup 后在 EMA21/55 执行区近侧挂被动限价单。",
        "- `trigger_confirm_next_open`: 等 trigger confirm 后，下一根开盘按当前系统方式入场。",
        "- `trigger_breakout`: 等 trigger confirm 后，做 trigger bar high/low 突破式入场。",
        "",
        "## 总览",
        "",
        "| method | episodes | filled | fill_rate_pct | avg_fill_bars | avg_entry_worse_r_vs_setup_market | avg_stop_distance_change_pct_vs_setup_market | plus_1r_hit_rate_pct | plus_2r_hit_rate_pct | stop_first_rate_pct | mean_mfe_r | mean_mae_r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in overall:
        lines.append(
            f"| {item['method']} | {item['episodes']} | {item['filled']} | {item['fill_rate_pct']} | {item['avg_fill_bars']} | {item['avg_entry_worse_r_vs_setup_market']} | {item['avg_stop_distance_change_pct_vs_setup_market']} | {item['plus_1r_hit_rate_pct']} | {item['plus_2r_hit_rate_pct']} | {item['stop_first_rate_pct']} | {item['mean_mfe_r']} | {item['mean_mae_r']} |"
        )

    lines.extend(
        [
            "",
            "## 按方向",
            "",
            "| side | method | episodes | filled | fill_rate_pct | avg_entry_worse_r_vs_setup_market | avg_stop_distance_change_pct_vs_setup_market | plus_1r_hit_rate_pct | plus_2r_hit_rate_pct | mean_mfe_r | mean_mae_r |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in by_side:
        lines.append(
            f"| {item['side']} | {item['method']} | {item['episodes']} | {item['filled']} | {item['fill_rate_pct']} | {item['avg_entry_worse_r_vs_setup_market']} | {item['avg_stop_distance_change_pct_vs_setup_market']} | {item['plus_1r_hit_rate_pct']} | {item['plus_2r_hit_rate_pct']} | {item['mean_mfe_r']} | {item['mean_mae_r']} |"
        )

    lines.extend(
        [
            "",
            "## 按趋势强度桶",
            "",
            "| trend_strength_bucket | side | method | episodes | filled | fill_rate_pct | avg_entry_worse_r_vs_setup_market | avg_stop_distance_change_pct_vs_setup_market | plus_1r_hit_rate_pct | plus_2r_hit_rate_pct | mean_mfe_r | mean_mae_r |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in by_bucket:
        lines.append(
            f"| {item['trend_strength_bucket']} | {item['side']} | {item['method']} | {item['episodes']} | {item['filled']} | {item['fill_rate_pct']} | {item['avg_entry_worse_r_vs_setup_market']} | {item['avg_stop_distance_change_pct_vs_setup_market']} | {item['plus_1r_hit_rate_pct']} | {item['plus_2r_hit_rate_pct']} | {item['mean_mfe_r']} | {item['mean_mae_r']} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging()

    now = datetime.now(timezone.utc)
    end = parse_date(args.end) if args.end else now
    start = parse_date(args.start) if args.start else end - timedelta(days=args.years * 365)

    service = BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(take_profit_mode="fixed_r", fixed_take_profit_r=2.0),
    )

    timeline = build_setup_timeline(
        service=service,
        exchange=args.exchange,
        market_type=args.market_type,
        symbol=args.symbol,
        strategy_profile=args.strategy_profile,
        start=start,
        end=end,
    )
    episodes = extract_setup_episodes(timeline)
    records = build_counterfactual_records(
        timeline=timeline,
        episodes=episodes,
        symbol=args.symbol,
        slippage_bps=service.assumptions.slippage_bps,
        horizon_bars=args.horizon_bars,
    )
    overall = summarize_records(records, ["method"])
    by_side = summarize_records(records, ["side", "method"])
    by_bucket = summarize_records(records, ["trend_strength_bucket", "side", "method"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "strategy_profile": args.strategy_profile,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "horizon_bars": int(args.horizon_bars),
        "episode_count": len(episodes),
        "overall": overall,
        "by_side": by_side,
        "by_trend_bucket": by_bucket,
    }

    output_dir = Path(args.output_dir) / args.strategy_profile
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records_path = output_dir / f"counterfactual_entries_{stamp}_records.csv"
    json_path = output_dir / f"counterfactual_entries_{stamp}.json"
    md_path = output_dir / f"counterfactual_entries_{stamp}.md"
    records.to_csv(records_path, index=False)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            symbol=args.symbol,
            strategy_profile=args.strategy_profile,
            start=start,
            end=end,
            horizon_bars=int(args.horizon_bars),
            episodes=episodes,
            overall=overall,
            by_side=by_side,
            by_bucket=by_bucket,
        ),
        encoding="utf-8",
    )

    print(f"Saved records CSV: {records_path}")
    print(f"Saved report JSON: {json_path}")
    print(f"Saved report Markdown: {md_path}")
    print(md_path.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()
