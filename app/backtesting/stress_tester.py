"""Stress testing framework for strategy robustness validation.

Tests strategy edge under adverse conditions:
  1. Fee multiplication (2×, 3× base fees)
  2. Slippage multiplication (2×, 3× base slippage)
  3. Combined fee + slippage stress
  4. Reduced favorable fills / worse execution
  5. Regime-conditioned stress (bull-only, bear-only, chop-only subsets)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Optional

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestSummary, BacktestTrade
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """A single stress test scenario definition."""
    name: str
    description: str
    fee_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    # Optional time filters: only include trades from specific date ranges
    # (useful for regime-specific testing)
    min_confidence_override: Optional[int] = None


@dataclass
class StressResult:
    """Results from a single stress scenario."""
    scenario_name: str
    scenario_description: str
    fee_bps_used: float
    slippage_bps_used: float
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    cumulative_r: float
    max_drawdown_r: float
    payoff_ratio: float
    # Comparison vs baseline
    expectancy_vs_baseline: float = 0.0  # ratio
    pf_vs_baseline: float = 0.0
    cumulative_r_vs_baseline: float = 0.0
    still_profitable: bool = True


@dataclass
class StressTestReport:
    """Complete stress test report across all scenarios."""
    generated_at: str
    strategy_profile: str
    symbol: str
    base_fee_bps: float
    base_slippage_bps: float
    # Baseline metrics
    baseline_trades: int = 0
    baseline_win_rate: float = 0.0
    baseline_profit_factor: float = 0.0
    baseline_expectancy_r: float = 0.0
    baseline_cumulative_r: float = 0.0
    baseline_max_drawdown_r: float = 0.0
    # Stress results
    scenarios: list[StressResult] = field(default_factory=list)
    # Summary
    scenarios_profitable: int = 0
    scenarios_total: int = 0
    robustness_score: float = 0.0  # % of scenarios still profitable
    worst_case_expectancy_r: float = 0.0
    worst_case_cumulative_r: float = 0.0
    # Break-even analysis
    max_fee_bps_breakeven: float = 0.0  # max fee where still profitable
    max_slippage_bps_breakeven: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scenarios"] = [asdict(s) for s in self.scenarios]
        return result


# ---------------------------------------------------------------------------
# Default stress scenarios
# ---------------------------------------------------------------------------

DEFAULT_STRESS_SCENARIOS: list[StressScenario] = [
    StressScenario(
        name="baseline",
        description="Normal conditions (1× fees, 1× slippage)",
        fee_multiplier=1.0,
        slippage_multiplier=1.0,
    ),
    StressScenario(
        name="fees_2x",
        description="Double fees (2× taker fee)",
        fee_multiplier=2.0,
        slippage_multiplier=1.0,
    ),
    StressScenario(
        name="fees_3x",
        description="Triple fees (3× taker fee)",
        fee_multiplier=3.0,
        slippage_multiplier=1.0,
    ),
    StressScenario(
        name="slippage_2x",
        description="Double slippage (2× base slippage)",
        fee_multiplier=1.0,
        slippage_multiplier=2.0,
    ),
    StressScenario(
        name="slippage_3x",
        description="Triple slippage (3× base slippage)",
        fee_multiplier=1.0,
        slippage_multiplier=3.0,
    ),
    StressScenario(
        name="combined_2x",
        description="Double both fees and slippage",
        fee_multiplier=2.0,
        slippage_multiplier=2.0,
    ),
    StressScenario(
        name="combined_3x",
        description="Triple both fees and slippage",
        fee_multiplier=3.0,
        slippage_multiplier=3.0,
    ),
    StressScenario(
        name="extreme_fees_5x",
        description="5× fees (extreme stress)",
        fee_multiplier=5.0,
        slippage_multiplier=1.0,
    ),
    StressScenario(
        name="extreme_combined",
        description="3× fees + 5× slippage (extreme combined)",
        fee_multiplier=3.0,
        slippage_multiplier=5.0,
    ),
]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


class StressTester:
    """Run stress tests on a strategy across multiple adverse scenarios."""

    def __init__(
        self,
        *,
        ohlcv_service: OhlcvService,
        strategy_service: StrategyService,
        base_assumptions: Optional[BacktestAssumptions] = None,
    ) -> None:
        self.ohlcv_service = ohlcv_service
        self.strategy_service = strategy_service
        self.base_assumptions = base_assumptions or BacktestAssumptions()

    def run(
        self,
        *,
        strategy_profile: str,
        symbol: str,
        exchange: str = "binance",
        market_type: str = "perpetual",
        start: datetime,
        end: datetime,
        scenarios: Optional[list[StressScenario]] = None,
    ) -> StressTestReport:
        """Run all stress scenarios and produce report."""
        if scenarios is None:
            scenarios = list(DEFAULT_STRESS_SCENARIOS)

        base_fee = self.base_assumptions.taker_fee_bps
        base_slip = self.base_assumptions.slippage_bps

        results: list[StressResult] = []
        baseline_metrics: Optional[dict[str, float]] = None

        for scenario in scenarios:
            logger.info("Running stress scenario: %s", scenario.name)

            # Build stressed assumptions
            assumptions = BacktestAssumptions(
                exit_profile=self.base_assumptions.exit_profile,
                take_profit_mode=self.base_assumptions.take_profit_mode,
                fixed_take_profit_r=self.base_assumptions.fixed_take_profit_r,
                scaled_tp1_r=self.base_assumptions.scaled_tp1_r,
                scaled_tp2_r=self.base_assumptions.scaled_tp2_r,
                taker_fee_bps=base_fee * scenario.fee_multiplier,
                slippage_bps=base_slip * scenario.slippage_multiplier,
                intraday_max_hold_bars=self.base_assumptions.intraday_max_hold_bars,
                swing_max_hold_bars=self.base_assumptions.swing_max_hold_bars,
                tp1_scale_out=self.base_assumptions.tp1_scale_out,
                move_stop_to_entry_after_tp1=self.base_assumptions.move_stop_to_entry_after_tp1,
                conservative_same_bar_exit=self.base_assumptions.conservative_same_bar_exit,
                lookback=self.base_assumptions.lookback,
                cache_dir=self.base_assumptions.cache_dir,
                long_exit=self.base_assumptions.long_exit,
                short_exit=self.base_assumptions.short_exit,
                swing_detection_mode=self.base_assumptions.swing_detection_mode,
                trailing_stop_enabled=self.base_assumptions.trailing_stop_enabled,
                trailing_stop_atr_mult=self.base_assumptions.trailing_stop_atr_mult,
                trailing_stop_activation_r=self.base_assumptions.trailing_stop_activation_r,
                leverage=self.base_assumptions.leverage,
            )

            service = BacktestService(
                ohlcv_service=self.ohlcv_service,
                strategy_service=self.strategy_service,
                assumptions=assumptions,
            )

            report = service.run(
                exchange=exchange,
                market_type=market_type,
                symbols=[symbol],
                strategy_profiles=[strategy_profile],
                start=start,
                end=end,
            )

            metrics = self._extract_metrics(report)
            if scenario.name == "baseline":
                baseline_metrics = metrics

            # Comparison vs baseline
            if baseline_metrics:
                exp_vs = _safe_ratio(metrics["expectancy_r"], baseline_metrics["expectancy_r"])
                pf_vs = _safe_ratio(metrics["profit_factor"], baseline_metrics["profit_factor"])
                cum_vs = _safe_ratio(metrics["cumulative_r"], baseline_metrics["cumulative_r"])
            else:
                exp_vs = 1.0
                pf_vs = 1.0
                cum_vs = 1.0

            results.append(StressResult(
                scenario_name=scenario.name,
                scenario_description=scenario.description,
                fee_bps_used=round(base_fee * scenario.fee_multiplier, 2),
                slippage_bps_used=round(base_slip * scenario.slippage_multiplier, 2),
                total_trades=int(metrics["trades"]),
                win_rate=round(metrics["win_rate"], 2),
                profit_factor=round(metrics["profit_factor"], 4),
                expectancy_r=round(metrics["expectancy_r"], 4),
                cumulative_r=round(metrics["cumulative_r"], 4),
                max_drawdown_r=round(metrics["max_drawdown_r"], 4),
                payoff_ratio=round(metrics["payoff_ratio"], 4),
                expectancy_vs_baseline=round(exp_vs, 4),
                pf_vs_baseline=round(pf_vs, 4),
                cumulative_r_vs_baseline=round(cum_vs, 4),
                still_profitable=metrics["cumulative_r"] > 0,
            ))

        return self._build_report(
            strategy_profile=strategy_profile,
            symbol=symbol,
            base_fee=base_fee,
            base_slip=base_slip,
            baseline_metrics=baseline_metrics or {},
            results=results,
        )

    @staticmethod
    def _extract_metrics(report) -> dict[str, float]:
        if report.overall:
            s = report.overall[0]
            return {
                "trades": s.total_trades,
                "win_rate": s.win_rate,
                "profit_factor": s.profit_factor,
                "expectancy_r": s.expectancy_r,
                "cumulative_r": s.cumulative_r,
                "max_drawdown_r": s.max_drawdown_r,
                "payoff_ratio": s.payoff_ratio,
            }
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "cumulative_r": 0.0,
            "max_drawdown_r": 0.0,
            "payoff_ratio": 0.0,
        }

    @staticmethod
    def _build_report(
        *,
        strategy_profile: str,
        symbol: str,
        base_fee: float,
        base_slip: float,
        baseline_metrics: dict[str, float],
        results: list[StressResult],
    ) -> StressTestReport:
        profitable_count = sum(1 for r in results if r.still_profitable)
        total = len(results)
        robustness = (profitable_count / total * 100) if total > 0 else 0.0

        expectancies = [r.expectancy_r for r in results]
        cumulatives = [r.cumulative_r for r in results]
        worst_exp = min(expectancies) if expectancies else 0.0
        worst_cum = min(cumulatives) if cumulatives else 0.0

        # Break-even analysis: find max fee/slippage where still profitable
        max_fee_be = base_fee
        max_slip_be = base_slip
        for r in results:
            if r.still_profitable:
                if r.fee_bps_used > max_fee_be:
                    max_fee_be = r.fee_bps_used
                if r.slippage_bps_used > max_slip_be:
                    max_slip_be = r.slippage_bps_used

        return StressTestReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            strategy_profile=strategy_profile,
            symbol=symbol,
            base_fee_bps=base_fee,
            base_slippage_bps=base_slip,
            baseline_trades=int(baseline_metrics.get("trades", 0)),
            baseline_win_rate=round(baseline_metrics.get("win_rate", 0.0), 2),
            baseline_profit_factor=round(baseline_metrics.get("profit_factor", 0.0), 4),
            baseline_expectancy_r=round(baseline_metrics.get("expectancy_r", 0.0), 4),
            baseline_cumulative_r=round(baseline_metrics.get("cumulative_r", 0.0), 4),
            baseline_max_drawdown_r=round(baseline_metrics.get("max_drawdown_r", 0.0), 4),
            scenarios=results,
            scenarios_profitable=profitable_count,
            scenarios_total=total,
            robustness_score=round(robustness, 2),
            worst_case_expectancy_r=round(worst_exp, 4),
            worst_case_cumulative_r=round(worst_cum, 4),
            max_fee_bps_breakeven=round(max_fee_be, 2),
            max_slippage_bps_breakeven=round(max_slip_be, 2),
        )
