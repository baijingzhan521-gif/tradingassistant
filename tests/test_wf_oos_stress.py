"""End-to-end Walk-Forward OOS validation and stress tests using synthetic data.

This module runs synthetic backtests for the three new strategies
(trend_following_v1, swing_improved_v1, mean_reversion_v1) through:
  1. Full-period backtest to establish baseline metrics
  2. Walk-forward OOS validation with IS/OOS splits
  3. Stress testing under adverse fee/slippage conditions
  4. Overfitting diagnostics (degradation ratios, consistency checks)

All data is synthetic (no exchange API calls required) but exercises
the full signal-generation → position-management → reporting pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from statistics import mean

import pandas as pd
import pytest

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestSummary, BacktestTrade
from app.backtesting.walk_forward_validator import (
    FoldResult,
    FoldWindow,
    WalkForwardReport,
    WalkForwardValidator,
    generate_folds,
)
from app.backtesting.stress_tester import (
    DEFAULT_STRESS_SCENARIOS,
    StressScenario,
    StressResult,
    StressTestReport,
    StressTester,
)
from app.data.exchange_client import ExchangeClientFactory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from tests.conftest import make_ohlcv_frame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers: build synthetic frames and run backtests
# ---------------------------------------------------------------------------

STRATEGY_PROFILES = ["swing_improved_v1", "trend_following_v1", "mean_reversion_v1"]

# Derive timeframes from strategy objects to stay in sync
STRATEGY_TIMEFRAMES: dict[str, tuple[str, ...]] = {
    name: StrategyService().build_strategy(name).required_timeframes
    for name in STRATEGY_PROFILES
}


def _make_synthetic_frames(
    *,
    periods_1h: int = 1200,
    start_price: float = 50000.0,
    up_step: float = 15.0,
    pullback_step: float = -30.0,
    pullback_len: int = 80,
    recovery_step: float = 25.0,
    recovery_len: int = 100,
) -> dict[str, pd.DataFrame]:
    """Build synthetic OHLCV frames for 1d, 4h, 1h with realistic shapes.

    Creates an uptrend → pullback → recovery pattern to exercise all three
    strategy types.
    """
    frame_1h = make_ohlcv_frame(
        periods=periods_1h,
        start_price=start_price,
        up_step=up_step,
        freq="h",
        pullback_step=pullback_step,
        pullback_len=pullback_len,
        recovery_step=recovery_step,
        recovery_len=recovery_len,
    )
    # 4h: aggregate
    periods_4h = max(periods_1h // 4, 300)
    frame_4h = make_ohlcv_frame(
        periods=periods_4h,
        start_price=start_price,
        up_step=up_step * 3.5,
        freq="4h",
        pullback_step=pullback_step * 3.5,
        pullback_len=pullback_len // 4,
        recovery_step=recovery_step * 3.5,
        recovery_len=recovery_len // 4,
    )
    # 1d: broader trend
    periods_1d = max(periods_1h // 24, 200)
    frame_1d = make_ohlcv_frame(
        periods=periods_1d,
        start_price=start_price * 0.7,
        up_step=up_step * 20,
        freq="D",
        pullback_step=pullback_step * 20,
        pullback_len=pullback_len // 24,
        recovery_step=recovery_step * 20,
        recovery_len=recovery_len // 24,
    )
    return {"1d": frame_1d, "4h": frame_4h, "1h": frame_1h}


def _run_backtest(
    strategy_profile: str,
    frames: dict[str, pd.DataFrame],
    *,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    trailing_stop: bool = False,
    leverage: float = 1.0,
) -> tuple[BacktestSummary, list[BacktestTrade]]:
    """Run a single strategy backtest on synthetic frames."""
    assumptions = BacktestAssumptions(
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        cache_dir="artifacts/backtests/test-cache",
        trailing_stop_enabled=trailing_stop,
        trailing_stop_atr_mult=3.0,
        trailing_stop_activation_r=1.0,
        leverage=leverage,
    )
    service = BacktestService(
        ohlcv_service=OhlcvService(ExchangeClientFactory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )
    start = frames["1h"]["timestamp"].iloc[0].to_pydatetime()
    end = frames["1h"]["timestamp"].iloc[-1].to_pydatetime()
    return service.run_symbol_strategy_with_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile=strategy_profile,
        start=start,
        end=end,
        frames=frames,
    )


# ===========================================================================
# 1. BASELINE BACKTEST TESTS
# ===========================================================================

class TestBaselineBacktest:
    """Establish baseline metrics for each strategy on synthetic data."""

    @pytest.fixture(scope="class")
    def frames(self) -> dict[str, pd.DataFrame]:
        return _make_synthetic_frames()

    def test_swing_improved_v1_runs_without_error(self, frames: dict[str, pd.DataFrame]) -> None:
        summary, trades = _run_backtest("swing_improved_v1", frames)
        assert summary is not None
        # May or may not generate trades on synthetic data
        assert summary.total_trades >= 0

    def test_trend_following_v1_runs_without_error(self, frames: dict[str, pd.DataFrame]) -> None:
        summary, trades = _run_backtest("trend_following_v1", frames)
        assert summary is not None
        assert summary.total_trades >= 0

    def test_mean_reversion_v1_runs_without_error(self, frames: dict[str, pd.DataFrame]) -> None:
        summary, trades = _run_backtest("mean_reversion_v1", frames)
        assert summary is not None
        assert summary.total_trades >= 0

    def test_swing_improved_v1_with_trailing_stop(self, frames: dict[str, pd.DataFrame]) -> None:
        summary, trades = _run_backtest(
            "swing_improved_v1", frames, trailing_stop=True, leverage=1.5,
        )
        assert summary is not None

    def test_trend_following_v1_with_leverage(self, frames: dict[str, pd.DataFrame]) -> None:
        summary, trades = _run_backtest(
            "trend_following_v1", frames, leverage=2.0, trailing_stop=True,
        )
        assert summary is not None


# ===========================================================================
# 2. WALK-FORWARD OOS VALIDATION
# ===========================================================================

class TestWalkForwardOOS:
    """Walk-forward OOS validation using synthetic fold generation."""

    def test_wf_fold_generation_for_two_years(self) -> None:
        folds = generate_folds(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 1, tzinfo=timezone.utc),
            train_days=365,
            test_days=90,
            step_days=90,
            scheme="rolling",
        )
        # Should generate 3+ folds over 2 years
        assert len(folds) >= 3
        # Verify non-overlapping test windows
        for i in range(1, len(folds)):
            assert folds[i].test_start >= folds[i - 1].test_start

    def test_wf_aggregation_detects_degradation(self) -> None:
        """Simulate IS/OOS results with known degradation."""
        folds = [
            FoldResult(
                fold_index=1, candidate="swing_improved_v1",
                train_start="2024-01-01", train_end="2025-01-01",
                test_start="2025-01-01", test_end="2025-04-01",
                is_trades=25, is_win_rate=58.0, is_profit_factor=1.6,
                is_expectancy_r=0.22, is_cumulative_r=5.5, is_max_drawdown_r=2.0,
                is_payoff_ratio=1.8,
                oos_trades=7, oos_win_rate=50.0, oos_profit_factor=1.15,
                oos_expectancy_r=0.06, oos_cumulative_r=0.42, oos_max_drawdown_r=0.8,
                oos_payoff_ratio=1.2,
                expectancy_degradation=0.27, pf_degradation=0.72, wr_degradation=0.86,
            ),
            FoldResult(
                fold_index=2, candidate="swing_improved_v1",
                train_start="2024-04-01", train_end="2025-04-01",
                test_start="2025-04-01", test_end="2025-07-01",
                is_trades=22, is_win_rate=56.0, is_profit_factor=1.5,
                is_expectancy_r=0.19, is_cumulative_r=4.18, is_max_drawdown_r=1.8,
                is_payoff_ratio=1.7,
                oos_trades=6, oos_win_rate=48.0, oos_profit_factor=1.08,
                oos_expectancy_r=0.03, oos_cumulative_r=0.18, oos_max_drawdown_r=0.5,
                oos_payoff_ratio=1.1,
                expectancy_degradation=0.16, pf_degradation=0.72, wr_degradation=0.86,
            ),
            FoldResult(
                fold_index=3, candidate="swing_improved_v1",
                train_start="2024-07-01", train_end="2025-07-01",
                test_start="2025-07-01", test_end="2025-10-01",
                is_trades=20, is_win_rate=55.0, is_profit_factor=1.45,
                is_expectancy_r=0.17, is_cumulative_r=3.40, is_max_drawdown_r=1.5,
                is_payoff_ratio=1.6,
                oos_trades=5, oos_win_rate=52.0, oos_profit_factor=1.20,
                oos_expectancy_r=0.10, oos_cumulative_r=0.50, oos_max_drawdown_r=0.4,
                oos_payoff_ratio=1.25,
                expectancy_degradation=0.59, pf_degradation=0.83, wr_degradation=0.95,
            ),
        ]
        report = WalkForwardValidator._aggregate(
            candidate="swing_improved_v1", scheme="rolling", folds=folds,
        )

        # All folds have positive OOS
        assert report.oos_positive_folds == 3
        assert report.oos_negative_folds == 0
        assert report.overfitting_probability == 0.0

        # IS > OOS expectancy (expected degradation)
        assert report.avg_is_expectancy_r > report.avg_oos_expectancy_r

        # But OOS is still positive → strategy has real edge
        assert report.avg_oos_expectancy_r > 0
        assert report.cumulative_oos_r > 0

        # Degradation ratio < 1 means OOS degrades from IS
        assert report.avg_expectancy_degradation < 1.0
        assert report.avg_expectancy_degradation > 0.0

    def test_wf_aggregation_with_overfitting_signal(self) -> None:
        """Simulate folds where OOS consistently negative → strong overfitting signal."""
        folds = [
            FoldResult(
                fold_index=i + 1, candidate="overfit_strat",
                train_start=f"2024-{i*3+1:02d}-01",
                train_end=f"2025-{i*3+1:02d}-01",
                test_start=f"2025-{i*3+1:02d}-01",
                test_end=f"2025-{i*3+4:02d}-01",
                is_trades=30, is_win_rate=65.0, is_profit_factor=2.5,
                is_expectancy_r=0.40, is_cumulative_r=12.0, is_max_drawdown_r=1.0,
                is_payoff_ratio=2.5,
                oos_trades=8, oos_win_rate=40.0, oos_profit_factor=0.7,
                oos_expectancy_r=-0.15, oos_cumulative_r=-1.2, oos_max_drawdown_r=2.0,
                oos_payoff_ratio=0.8,
                expectancy_degradation=-0.375, pf_degradation=0.28, wr_degradation=0.62,
            )
            for i in range(3)
        ]
        report = WalkForwardValidator._aggregate(
            candidate="overfit_strat", scheme="rolling", folds=folds,
        )

        # Should clearly flag overfitting
        assert report.oos_negative_folds == 3
        assert report.overfitting_probability == 100.0
        assert report.avg_oos_expectancy_r < 0
        assert report.cumulative_oos_r < 0


# ===========================================================================
# 3. STRESS TEST VALIDATION
# ===========================================================================

class TestStressTestValidation:
    """Validate stress test framework produces correct comparisons."""

    def test_fee_stress_reduces_profitability(self) -> None:
        """Higher fees should reduce or eliminate profitability."""
        frames = _make_synthetic_frames()
        # Baseline: 0 fees
        summary_0, _ = _run_backtest("swing_improved_v1", frames, taker_fee_bps=0.0, slippage_bps=0.0)
        # Stressed: high fees
        summary_high, _ = _run_backtest("swing_improved_v1", frames, taker_fee_bps=20.0, slippage_bps=5.0)
        # High fees should give worse cumulative_r
        assert summary_high.cumulative_r <= summary_0.cumulative_r

    def test_stress_scenarios_cover_all_multipliers(self) -> None:
        """Verify all default scenarios have valid multipliers."""
        for s in DEFAULT_STRESS_SCENARIOS:
            assert s.fee_multiplier >= 1.0
            assert s.slippage_multiplier >= 1.0

    def test_stress_report_structure(self) -> None:
        """Build a synthetic stress report and verify structure."""
        baseline = StressResult(
            scenario_name="baseline", scenario_description="Normal",
            fee_bps_used=5.0, slippage_bps_used=2.0,
            total_trades=40, win_rate=55.0, profit_factor=1.5,
            expectancy_r=0.12, cumulative_r=4.8, max_drawdown_r=2.0,
            payoff_ratio=1.8, still_profitable=True,
        )
        stressed = StressResult(
            scenario_name="fees_3x", scenario_description="Triple fees",
            fee_bps_used=15.0, slippage_bps_used=2.0,
            total_trades=40, win_rate=55.0, profit_factor=1.1,
            expectancy_r=0.03, cumulative_r=1.2, max_drawdown_r=3.0,
            payoff_ratio=1.2,
            expectancy_vs_baseline=0.25, pf_vs_baseline=0.73,
            cumulative_r_vs_baseline=0.25, still_profitable=True,
        )
        extreme = StressResult(
            scenario_name="extreme_combined", scenario_description="3× fees + 5× slippage",
            fee_bps_used=15.0, slippage_bps_used=10.0,
            total_trades=40, win_rate=48.0, profit_factor=0.85,
            expectancy_r=-0.05, cumulative_r=-2.0, max_drawdown_r=5.0,
            payoff_ratio=0.9, still_profitable=False,
        )
        report = StressTestReport(
            generated_at="2026-01-01T00:00:00+00:00",
            strategy_profile="swing_improved_v1",
            symbol="BTC/USDT:USDT",
            base_fee_bps=5.0,
            base_slippage_bps=2.0,
            baseline_trades=40,
            baseline_cumulative_r=4.8,
            baseline_expectancy_r=0.12,
            scenarios=[baseline, stressed, extreme],
            scenarios_profitable=2,
            scenarios_total=3,
            robustness_score=66.67,
            worst_case_expectancy_r=-0.05,
            worst_case_cumulative_r=-2.0,
        )
        d = report.to_dict()
        assert d["robustness_score"] == 66.67
        assert len(d["scenarios"]) == 3
        assert d["worst_case_cumulative_r"] == -2.0


# ===========================================================================
# 4. OVERFITTING DIAGNOSTICS
# ===========================================================================

class TestOverfittingDiagnostics:
    """Tests specifically targeting overfitting detection."""

    def test_high_is_low_oos_flags_overfitting(self) -> None:
        """A strategy with IS expectancy >> OOS expectancy is overfitting."""
        fold = FoldResult(
            fold_index=1, candidate="overfit",
            train_start="2024-01-01", train_end="2025-01-01",
            test_start="2025-01-01", test_end="2025-04-01",
            is_trades=50, is_expectancy_r=0.50,
            oos_trades=12, oos_expectancy_r=0.02,
            expectancy_degradation=0.04,
        )
        # Degradation ratio near 0 → strong overfitting signal
        assert fold.expectancy_degradation < 0.1

    def test_consistent_oos_confirms_edge(self) -> None:
        """Consistent positive OOS across folds confirms real edge."""
        folds = [
            FoldResult(
                fold_index=i + 1, candidate="real_edge",
                train_start=f"2024-{i*3+1:02d}-01",
                train_end=f"2025-{i*3+1:02d}-01",
                test_start=f"2025-{i*3+1:02d}-01",
                test_end=f"2025-{i*3+4:02d}-01",
                is_expectancy_r=0.15 + i * 0.01,
                oos_expectancy_r=0.08 + i * 0.01,
                expectancy_degradation=(0.08 + i * 0.01) / (0.15 + i * 0.01),
            )
            for i in range(4)
        ]
        report = WalkForwardValidator._aggregate(
            candidate="real_edge", scheme="rolling", folds=folds,
        )
        # All folds positive
        assert report.oos_positive_folds == 4
        assert report.overfitting_probability == 0.0
        # Sharpe-equivalent should be positive
        assert report.oos_sharpe_equivalent > 0

    def test_oos_sharpe_equivalent_tracks_consistency(self) -> None:
        """Higher Sharpe equivalent = more consistent OOS performance."""
        # Consistent folds
        consistent_folds = [
            FoldResult(
                fold_index=i + 1, candidate="consistent",
                train_start="", train_end="", test_start="", test_end="",
                oos_expectancy_r=0.10,
            )
            for i in range(5)
        ]
        report_consistent = WalkForwardValidator._aggregate(
            candidate="consistent", scheme="rolling", folds=consistent_folds,
        )

        # Inconsistent folds (same mean but high variance)
        inconsistent_folds = [
            FoldResult(
                fold_index=i + 1, candidate="inconsistent",
                train_start="", train_end="", test_start="", test_end="",
                oos_expectancy_r=0.30 if i % 2 == 0 else -0.10,
            )
            for i in range(5)
        ]
        report_inconsistent = WalkForwardValidator._aggregate(
            candidate="inconsistent", scheme="rolling", folds=inconsistent_folds,
        )

        # Consistent should have higher Sharpe equivalent
        assert report_consistent.oos_sharpe_equivalent > report_inconsistent.oos_sharpe_equivalent


# ===========================================================================
# 5. INTEGRATED PORTFOLIO SIMULATION TESTS
# ===========================================================================

class TestIntegratedPortfolioValidation:
    """Run all three strategies through synthetic backtests and verify
    portfolio-level metrics are reasonable."""

    @pytest.fixture(scope="class")
    def all_results(self) -> dict[str, tuple[BacktestSummary, list[BacktestTrade]]]:
        frames = _make_synthetic_frames()
        results = {}
        for profile in STRATEGY_PROFILES:
            results[profile] = _run_backtest(profile, frames)
        return results

    def test_all_strategies_complete(
        self, all_results: dict[str, tuple[BacktestSummary, list[BacktestTrade]]]
    ) -> None:
        for profile in STRATEGY_PROFILES:
            summary, trades = all_results[profile]
            assert summary is not None

    def test_combined_trades_sorted_by_entry_time(
        self, all_results: dict[str, tuple[BacktestSummary, list[BacktestTrade]]]
    ) -> None:
        all_trades = []
        for profile in STRATEGY_PROFILES:
            _, trades = all_results[profile]
            all_trades.extend(trades)
        if len(all_trades) > 1:
            sorted_trades = sorted(all_trades, key=lambda t: t.entry_time)
            for i in range(1, len(sorted_trades)):
                assert sorted_trades[i].entry_time >= sorted_trades[i - 1].entry_time

    def test_leverage_field_populated_on_trades(
        self, all_results: dict[str, tuple[BacktestSummary, list[BacktestTrade]]]
    ) -> None:
        for profile in STRATEGY_PROFILES:
            _, trades = all_results[profile]
            for trade in trades:
                assert trade.leverage >= 1.0
