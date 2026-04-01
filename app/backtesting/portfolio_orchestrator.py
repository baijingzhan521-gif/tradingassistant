from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.backtesting.service import (
    BacktestAssumptions,
    BacktestService,
    BacktestSummary,
    BacktestTrade,
)
from app.data.ohlcv_service import OhlcvService
from app.indicators.regime_classifier import MarketRegimeType
from app.services.strategy_service import StrategyService

logger = logging.getLogger(__name__)


# Strategy allocation weights per regime
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    MarketRegimeType.BULL_TREND.value: {
        "trend_following_v1": 0.60,
        "swing_improved_v1": 0.30,
        "mean_reversion_v1": 0.10,
    },
    MarketRegimeType.BEAR_TREND.value: {
        "trend_following_v1": 0.50,
        "swing_improved_v1": 0.40,
        "mean_reversion_v1": 0.10,
    },
    MarketRegimeType.LOW_VOL_RANGE.value: {
        "trend_following_v1": 0.10,
        "swing_improved_v1": 0.20,
        "mean_reversion_v1": 0.70,
    },
    MarketRegimeType.HIGH_VOL_CHOP.value: {
        "trend_following_v1": 0.00,
        "swing_improved_v1": 0.10,
        "mean_reversion_v1": 0.30,
    },
    MarketRegimeType.TRANSITION.value: {
        "trend_following_v1": 0.20,
        "swing_improved_v1": 0.50,
        "mean_reversion_v1": 0.30,
    },
}

# Maximum leverage per regime
REGIME_MAX_LEVERAGE: dict[str, float] = {
    MarketRegimeType.BULL_TREND.value: 3.0,
    MarketRegimeType.BEAR_TREND.value: 2.0,
    MarketRegimeType.LOW_VOL_RANGE.value: 1.5,
    MarketRegimeType.HIGH_VOL_CHOP.value: 0.5,
    MarketRegimeType.TRANSITION.value: 1.0,
}


@dataclass
class PortfolioConfig:
    """Configuration for portfolio orchestration."""
    strategy_profiles: list[str] = field(
        default_factory=lambda: ["trend_following_v1", "swing_improved_v1", "mean_reversion_v1"]
    )
    regime_weights: dict[str, dict[str, float]] = field(default_factory=lambda: dict(REGIME_WEIGHTS))
    regime_max_leverage: dict[str, float] = field(default_factory=lambda: dict(REGIME_MAX_LEVERAGE))
    # Drawdown circuit breaker levels (percentage of peak equity)
    drawdown_level1_pct: float = 10.0
    drawdown_level2_pct: float = 20.0
    drawdown_level3_pct: float = 25.0
    drawdown_cooldown_bars: int = 168  # 7 days on 1h bars
    # Dynamic leverage parameters
    base_leverage: float = 1.0
    max_leverage: float = 3.0
    normal_vol: float = 0.5  # "normal" annualized vol baseline


@dataclass
class PortfolioState:
    """Tracks portfolio equity and drawdown state."""
    initial_equity: float = 10000.0
    equity: float = 10000.0
    peak_equity: float = 10000.0
    current_drawdown_pct: float = 0.0
    drawdown_level: int = 0  # 0=normal, 1-3=circuit breaker levels
    cooldown_remaining: int = 0
    leverage_multiplier: float = 1.0
    current_regime: str = MarketRegimeType.TRANSITION.value


@dataclass
class PortfolioReport:
    """Report combining results from all strategies."""
    generated_at: str
    exchange: str
    market_type: str
    start: str
    end: str
    symbols: list[str]
    config: dict[str, Any]
    # Per-strategy results
    strategy_summaries: list[BacktestSummary]
    strategy_trades: dict[str, list[BacktestTrade]]
    # Combined portfolio metrics
    total_trades: int
    combined_cumulative_r: float
    combined_cumulative_return_pct: float
    combined_max_drawdown_r: float
    combined_win_rate: float
    combined_profit_factor: float
    combined_expectancy_r: float
    # Regime analysis
    regime_distribution: dict[str, int]
    regime_pnl: dict[str, float]
    # Equity curve data
    equity_curve: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "start": self.start,
            "end": self.end,
            "symbols": self.symbols,
            "config": self.config,
            "strategy_summaries": [asdict(s) for s in self.strategy_summaries],
            "strategy_trades": {
                k: [asdict(t) for t in v] for k, v in self.strategy_trades.items()
            },
            "total_trades": self.total_trades,
            "combined_cumulative_r": self.combined_cumulative_r,
            "combined_cumulative_return_pct": self.combined_cumulative_return_pct,
            "combined_max_drawdown_r": self.combined_max_drawdown_r,
            "combined_win_rate": self.combined_win_rate,
            "combined_profit_factor": self.combined_profit_factor,
            "combined_expectancy_r": self.combined_expectancy_r,
            "regime_distribution": self.regime_distribution,
            "regime_pnl": self.regime_pnl,
            "equity_curve": self.equity_curve,
        }


def compute_dynamic_leverage(
    *,
    regime: str,
    confidence: float,
    current_drawdown_pct: float,
    realized_vol: float,
    config: PortfolioConfig,
) -> float:
    """Compute target leverage from multi-factor model.

    Factors:
    1. Base leverage from regime
    2. Confidence adjustment (0.5-1.0)
    3. Drawdown brake (reduces as DD increases)
    4. Volatility adjustment (inverse relationship)
    """
    base = config.regime_max_leverage.get(regime, config.base_leverage)

    # Confidence: scale from 0.5 at min to 1.0 at max
    confidence_mult = 0.5 + (min(confidence, 100) / 100) * 0.5

    # Drawdown brake: linear reduction, hits 0.2 at level3
    max_dd = config.drawdown_level3_pct
    dd_mult = max(0.2, 1.0 - current_drawdown_pct / max_dd) if max_dd > 0 else 1.0

    # Volatility: inverse relationship
    normal_vol = config.normal_vol if config.normal_vol > 0 else 0.5
    vol_mult = min(1.0, normal_vol / realized_vol) if realized_vol > 0 else 1.0

    target = base * confidence_mult * dd_mult * vol_mult
    return round(min(max(target, 0.0), config.max_leverage), 4)


def apply_drawdown_circuit_breaker(
    state: PortfolioState,
    config: PortfolioConfig,
) -> PortfolioState:
    """Update drawdown level and cooldown based on current state."""
    if state.equity > state.peak_equity:
        state.peak_equity = state.equity

    if state.peak_equity > 0:
        state.current_drawdown_pct = ((state.peak_equity - state.equity) / state.peak_equity) * 100
    else:
        state.current_drawdown_pct = 0.0

    # Determine drawdown level
    if state.current_drawdown_pct >= config.drawdown_level3_pct:
        if state.drawdown_level < 3:
            state.drawdown_level = 3
            state.cooldown_remaining = config.drawdown_cooldown_bars
            logger.warning(
                "Circuit breaker LEVEL 3: DD=%.1f%%, entering full cooldown (%d bars)",
                state.current_drawdown_pct, config.drawdown_cooldown_bars,
            )
    elif state.current_drawdown_pct >= config.drawdown_level2_pct:
        if state.drawdown_level < 2:
            state.drawdown_level = 2
            logger.warning("Circuit breaker LEVEL 2: DD=%.1f%%", state.current_drawdown_pct)
    elif state.current_drawdown_pct >= config.drawdown_level1_pct:
        if state.drawdown_level < 1:
            state.drawdown_level = 1
            logger.info("Circuit breaker LEVEL 1: DD=%.1f%%", state.current_drawdown_pct)

    # Recovery
    if state.current_drawdown_pct < config.drawdown_level1_pct and state.drawdown_level > 0:
        if state.cooldown_remaining <= 0:
            state.drawdown_level = 0
            logger.info("Circuit breaker RECOVERED: equity near peak")

    # Cooldown tick
    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1

    # Leverage multiplier based on drawdown level
    if state.drawdown_level == 0:
        state.leverage_multiplier = 1.0
    elif state.drawdown_level == 1:
        state.leverage_multiplier = 0.7
    elif state.drawdown_level == 2:
        state.leverage_multiplier = 0.4
    else:  # level 3
        state.leverage_multiplier = 0.0  # Full stop

    return state


class PortfolioOrchestrator:
    """Runs multiple strategies as a portfolio with regime-based allocation."""

    def __init__(
        self,
        *,
        ohlcv_service: OhlcvService,
        strategy_service: StrategyService,
        config: Optional[PortfolioConfig] = None,
        base_assumptions: Optional[BacktestAssumptions] = None,
    ) -> None:
        self.ohlcv_service = ohlcv_service
        self.strategy_service = strategy_service
        self.config = config or PortfolioConfig()
        self.base_assumptions = base_assumptions or BacktestAssumptions()

    def run(
        self,
        *,
        exchange: str,
        market_type: str,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> PortfolioReport:
        """Run all strategies and combine results into a portfolio report."""
        strategy_summaries: list[BacktestSummary] = []
        strategy_trades: dict[str, list[BacktestTrade]] = {}
        all_trades: list[BacktestTrade] = []

        for profile in self.config.strategy_profiles:
            # Each strategy gets its own assumptions (can customize per-strategy)
            assumptions = self._build_strategy_assumptions(profile)
            service = BacktestService(
                ohlcv_service=self.ohlcv_service,
                strategy_service=self.strategy_service,
                assumptions=assumptions,
            )

            report = service.run(
                exchange=exchange,
                market_type=market_type,
                symbols=symbols,
                strategy_profiles=[profile],
                start=start,
                end=end,
            )

            strategy_summaries.extend(report.overall)
            strategy_trades[profile] = report.trades
            all_trades.extend(report.trades)

        # Combine and analyze
        combined = self._combine_portfolio(all_trades, strategy_trades)

        return PortfolioReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            exchange=exchange,
            market_type=market_type,
            start=start.isoformat(),
            end=end.isoformat(),
            symbols=symbols,
            config=asdict(self.config),
            strategy_summaries=strategy_summaries,
            strategy_trades=strategy_trades,
            **combined,
        )

    def _build_strategy_assumptions(self, profile: str) -> BacktestAssumptions:
        """Build per-strategy assumptions. Trend following gets trailing stop + leverage."""
        base = self.base_assumptions
        if profile == "trend_following_v1":
            return BacktestAssumptions(
                exit_profile="trend_following_trailing",
                take_profit_mode="scaled",
                scaled_tp1_r=1.0,
                scaled_tp2_r=5.0,
                taker_fee_bps=base.taker_fee_bps,
                slippage_bps=base.slippage_bps,
                tp1_scale_out=0.3,
                move_stop_to_entry_after_tp1=True,
                swing_max_hold_bars=480,  # 20 days on 1h
                lookback=base.lookback,
                cache_dir=base.cache_dir,
                swing_detection_mode=base.swing_detection_mode,
                trailing_stop_enabled=True,
                trailing_stop_atr_mult=3.0,
                trailing_stop_activation_r=1.0,
                leverage=2.0,
            )
        if profile == "mean_reversion_v1":
            return BacktestAssumptions(
                exit_profile="mean_reversion_fixed",
                take_profit_mode="fixed_r",
                fixed_take_profit_r=1.0,
                taker_fee_bps=base.taker_fee_bps,
                slippage_bps=base.slippage_bps,
                swing_max_hold_bars=48,  # 2 days
                lookback=base.lookback,
                cache_dir=base.cache_dir,
                swing_detection_mode=base.swing_detection_mode,
                leverage=1.0,
            )
        # swing_improved_v1 and others: use base with slight modifications
        return BacktestAssumptions(
            exit_profile=base.exit_profile,
            take_profit_mode=base.take_profit_mode,
            scaled_tp1_r=1.0,
            scaled_tp2_r=3.0,
            taker_fee_bps=base.taker_fee_bps,
            slippage_bps=base.slippage_bps,
            tp1_scale_out=0.3,
            move_stop_to_entry_after_tp1=True,
            swing_max_hold_bars=base.swing_max_hold_bars,
            lookback=base.lookback,
            cache_dir=base.cache_dir,
            swing_detection_mode=base.swing_detection_mode,
            trailing_stop_enabled=True,
            trailing_stop_atr_mult=2.5,
            trailing_stop_activation_r=1.5,
            leverage=1.5,
        )

    def _combine_portfolio(
        self,
        all_trades: list[BacktestTrade],
        strategy_trades: dict[str, list[BacktestTrade]],
    ) -> dict[str, Any]:
        """Combine trades from all strategies into portfolio metrics."""
        if not all_trades:
            return {
                "total_trades": 0,
                "combined_cumulative_r": 0.0,
                "combined_cumulative_return_pct": 0.0,
                "combined_max_drawdown_r": 0.0,
                "combined_win_rate": 0.0,
                "combined_profit_factor": 0.0,
                "combined_expectancy_r": 0.0,
                "regime_distribution": {},
                "regime_pnl": {},
                "equity_curve": [],
            }

        # Sort all trades by entry time for combined equity curve
        sorted_trades = sorted(all_trades, key=lambda t: t.entry_time)
        pnl_rs = [t.pnl_r for t in sorted_trades]

        winners = [r for r in pnl_rs if r > 0]
        losers = [r for r in pnl_rs if r < 0]

        cumulative_r = sum(pnl_rs)
        cumulative_pct = sum(t.pnl_pct for t in sorted_trades)

        # Equity curve (R-based)
        equity_curve = list(np.cumsum(pnl_rs))
        running_peak = np.maximum.accumulate(equity_curve)
        drawdowns = running_peak - np.array(equity_curve)
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        win_rate = (len(winners) / len(sorted_trades) * 100) if sorted_trades else 0.0
        profit_factor = (sum(winners) / abs(sum(losers))) if losers else 0.0
        expectancy_r = mean(pnl_rs) if pnl_rs else 0.0

        # Per-strategy attribution
        regime_distribution: dict[str, int] = {}
        regime_pnl: dict[str, float] = {}
        for profile, trades in strategy_trades.items():
            regime_distribution[profile] = len(trades)
            regime_pnl[profile] = round(sum(t.pnl_r for t in trades), 4)

        return {
            "total_trades": len(sorted_trades),
            "combined_cumulative_r": round(cumulative_r, 4),
            "combined_cumulative_return_pct": round(cumulative_pct, 4),
            "combined_max_drawdown_r": round(max_dd, 4),
            "combined_win_rate": round(win_rate, 2),
            "combined_profit_factor": round(profit_factor, 4),
            "combined_expectancy_r": round(expectancy_r, 4),
            "regime_distribution": regime_distribution,
            "regime_pnl": regime_pnl,
            "equity_curve": [round(v, 4) for v in equity_curve],
        }
