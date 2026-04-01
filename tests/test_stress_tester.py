"""Tests for the stress testing framework."""
from __future__ import annotations

import pytest

from app.backtesting.stress_tester import (
    DEFAULT_STRESS_SCENARIOS,
    StressResult,
    StressScenario,
    StressTestReport,
    StressTester,
)
from app.utils.math_utils import safe_ratio as _safe_ratio
from app.backtesting.service import BacktestAssumptions


class TestStressScenario:
    def test_default_scenarios_exist(self) -> None:
        assert len(DEFAULT_STRESS_SCENARIOS) >= 5

    def test_baseline_is_first(self) -> None:
        assert DEFAULT_STRESS_SCENARIOS[0].name == "baseline"
        assert DEFAULT_STRESS_SCENARIOS[0].fee_multiplier == 1.0
        assert DEFAULT_STRESS_SCENARIOS[0].slippage_multiplier == 1.0

    def test_fee_multipliers_are_progressive(self) -> None:
        fee_scenarios = [s for s in DEFAULT_STRESS_SCENARIOS if "fees" in s.name]
        for s in fee_scenarios:
            assert s.fee_multiplier >= 1.0

    def test_combined_scenarios_stress_both(self) -> None:
        combined = [s for s in DEFAULT_STRESS_SCENARIOS if "combined" in s.name]
        for s in combined:
            assert s.fee_multiplier >= 2.0
            assert s.slippage_multiplier >= 2.0


class TestStressResult:
    def test_result_creation(self) -> None:
        r = StressResult(
            scenario_name="fees_2x",
            scenario_description="Double fees",
            fee_bps_used=10.0,
            slippage_bps_used=2.0,
            total_trades=50,
            win_rate=52.0,
            profit_factor=1.3,
            expectancy_r=0.08,
            cumulative_r=4.0,
            max_drawdown_r=2.0,
            payoff_ratio=1.5,
        )
        assert r.scenario_name == "fees_2x"
        assert r.still_profitable is True

    def test_unprofitable_result(self) -> None:
        r = StressResult(
            scenario_name="extreme",
            scenario_description="Extreme stress",
            fee_bps_used=25.0,
            slippage_bps_used=10.0,
            total_trades=50,
            win_rate=45.0,
            profit_factor=0.8,
            expectancy_r=-0.05,
            cumulative_r=-2.5,
            max_drawdown_r=5.0,
            payoff_ratio=1.0,
            still_profitable=False,
        )
        assert r.still_profitable is False
        assert r.cumulative_r < 0


class TestStressTestReport:
    def test_report_to_dict(self) -> None:
        report = StressTestReport(
            generated_at="2026-01-01T00:00:00+00:00",
            strategy_profile="test_strat",
            symbol="BTC/USDT:USDT",
            base_fee_bps=5.0,
            base_slippage_bps=2.0,
            baseline_trades=50,
            baseline_cumulative_r=5.0,
            scenarios=[
                StressResult(
                    scenario_name="baseline",
                    scenario_description="Normal",
                    fee_bps_used=5.0,
                    slippage_bps_used=2.0,
                    total_trades=50,
                    win_rate=55.0,
                    profit_factor=1.5,
                    expectancy_r=0.10,
                    cumulative_r=5.0,
                    max_drawdown_r=2.0,
                    payoff_ratio=1.8,
                ),
            ],
            scenarios_profitable=1,
            scenarios_total=1,
            robustness_score=100.0,
        )
        d = report.to_dict()
        assert d["strategy_profile"] == "test_strat"
        assert len(d["scenarios"]) == 1
        assert d["robustness_score"] == 100.0

    def test_robustness_calculation(self) -> None:
        # Construct a report with mixed profitability
        scenarios = [
            StressResult(
                scenario_name=f"s{i}",
                scenario_description=f"Scenario {i}",
                fee_bps_used=5.0 * (i + 1),
                slippage_bps_used=2.0,
                total_trades=30,
                win_rate=50.0,
                profit_factor=1.0 if i < 5 else 0.5,
                expectancy_r=0.05 if i < 5 else -0.10,
                cumulative_r=1.5 if i < 5 else -3.0,
                max_drawdown_r=2.0,
                payoff_ratio=1.0,
                still_profitable=i < 5,
            )
            for i in range(10)
        ]
        report = StressTestReport(
            generated_at="2026-01-01T00:00:00+00:00",
            strategy_profile="test",
            symbol="BTC/USDT:USDT",
            base_fee_bps=5.0,
            base_slippage_bps=2.0,
            scenarios=scenarios,
            scenarios_profitable=5,
            scenarios_total=10,
            robustness_score=50.0,
        )
        assert report.robustness_score == 50.0


class TestSafeRatio:
    def test_normal(self) -> None:
        assert _safe_ratio(10.0, 2.0) == 5.0

    def test_zero_denom(self) -> None:
        assert _safe_ratio(10.0, 0.0) == 0.0


class TestStressTesterBuild:
    """Test StressTester construction and scenario building."""

    def test_default_assumptions(self) -> None:
        from app.data.exchange_client import ExchangeClientFactory
        from app.data.ohlcv_service import OhlcvService
        from app.services.strategy_service import StrategyService

        tester = StressTester(
            ohlcv_service=OhlcvService(ExchangeClientFactory()),
            strategy_service=StrategyService(),
        )
        assert tester.base_assumptions.taker_fee_bps == 5.0
        assert tester.base_assumptions.slippage_bps == 2.0

    def test_custom_assumptions(self) -> None:
        from app.data.exchange_client import ExchangeClientFactory
        from app.data.ohlcv_service import OhlcvService
        from app.services.strategy_service import StrategyService

        custom = BacktestAssumptions(taker_fee_bps=7.0, slippage_bps=3.0)
        tester = StressTester(
            ohlcv_service=OhlcvService(ExchangeClientFactory()),
            strategy_service=StrategyService(),
            base_assumptions=custom,
        )
        assert tester.base_assumptions.taker_fee_bps == 7.0
        assert tester.base_assumptions.slippage_bps == 3.0
