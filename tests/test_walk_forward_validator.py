"""Tests for Walk-Forward OOS validation engine."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.backtesting.walk_forward_validator import (
    FoldResult,
    FoldWindow,
    WalkForwardReport,
    WalkForwardValidator,
    generate_folds,
    _safe_ratio,
)


class TestFoldGeneration:
    """Test walk-forward fold window generation."""

    def test_rolling_scheme_generates_expected_folds(self) -> None:
        folds = generate_folds(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 1, tzinfo=timezone.utc),
            train_days=365,
            test_days=90,
            step_days=90,
            scheme="rolling",
        )
        assert len(folds) >= 2
        # All folds should have index starting at 1
        assert folds[0].index == 1
        # Train comes before test
        for f in folds:
            assert f.train_start < f.train_end
            assert f.train_end == f.test_start
            assert f.test_start < f.test_end

    def test_anchored_scheme_expands_training(self) -> None:
        folds = generate_folds(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 1, tzinfo=timezone.utc),
            train_days=365,
            test_days=90,
            step_days=90,
            scheme="anchored",
        )
        assert len(folds) >= 2
        # All anchored folds start from the same point
        for f in folds:
            assert f.train_start == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_no_folds_if_insufficient_data(self) -> None:
        folds = generate_folds(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            train_days=365,
            test_days=90,
            step_days=90,
            scheme="rolling",
        )
        assert len(folds) == 0

    def test_single_fold_possible(self) -> None:
        folds = generate_folds(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 7, 1, tzinfo=timezone.utc),
            train_days=365,
            test_days=90,
            step_days=90,
            scheme="rolling",
        )
        assert len(folds) >= 1

    def test_folds_dont_exceed_end(self) -> None:
        folds = generate_folds(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 1, tzinfo=timezone.utc),
            train_days=365,
            test_days=90,
            step_days=90,
            scheme="rolling",
        )
        for f in folds:
            assert f.test_end <= datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestSafeRatio:
    def test_normal_division(self) -> None:
        assert _safe_ratio(10.0, 5.0) == 2.0

    def test_zero_denominator(self) -> None:
        assert _safe_ratio(10.0, 0.0) == 0.0

    def test_zero_numerator(self) -> None:
        assert _safe_ratio(0.0, 5.0) == 0.0


class TestFoldResult:
    def test_fold_result_creation(self) -> None:
        r = FoldResult(
            fold_index=1,
            candidate="test_strategy",
            train_start="2024-01-01",
            train_end="2025-01-01",
            test_start="2025-01-01",
            test_end="2025-04-01",
            is_trades=20,
            is_expectancy_r=0.15,
            oos_trades=5,
            oos_expectancy_r=0.10,
        )
        assert r.fold_index == 1
        assert r.is_trades == 20
        assert r.oos_trades == 5

    def test_degradation_defaults(self) -> None:
        r = FoldResult(
            fold_index=1,
            candidate="test",
            train_start="2024-01-01",
            train_end="2025-01-01",
            test_start="2025-01-01",
            test_end="2025-04-01",
        )
        assert r.expectancy_degradation == 0.0
        assert r.pf_degradation == 0.0


class TestWalkForwardReport:
    def test_empty_report(self) -> None:
        report = WalkForwardReport(
            generated_at="2026-01-01T00:00:00+00:00",
            candidate="test",
            scheme="rolling",
            num_folds=0,
            total_is_trades=0,
            total_oos_trades=0,
        )
        assert report.num_folds == 0
        assert report.overfitting_probability == 0.0

    def test_to_dict(self) -> None:
        report = WalkForwardReport(
            generated_at="2026-01-01T00:00:00+00:00",
            candidate="test",
            scheme="rolling",
            num_folds=1,
            total_is_trades=10,
            total_oos_trades=3,
            folds=[
                FoldResult(
                    fold_index=1,
                    candidate="test",
                    train_start="2024-01-01",
                    train_end="2025-01-01",
                    test_start="2025-01-01",
                    test_end="2025-04-01",
                )
            ],
        )
        d = report.to_dict()
        assert d["num_folds"] == 1
        assert len(d["folds"]) == 1
        assert d["folds"][0]["fold_index"] == 1


class TestWalkForwardValidatorAggregation:
    """Test the static _aggregate method directly with synthetic data."""

    def test_aggregate_positive_folds(self) -> None:
        folds = [
            FoldResult(
                fold_index=1,
                candidate="strat_a",
                train_start="2024-01-01", train_end="2025-01-01",
                test_start="2025-01-01", test_end="2025-04-01",
                is_trades=20, is_expectancy_r=0.20, is_profit_factor=1.5, is_win_rate=55.0,
                is_cumulative_r=4.0, is_max_drawdown_r=2.0, is_payoff_ratio=1.8,
                oos_trades=6, oos_expectancy_r=0.12, oos_profit_factor=1.2, oos_win_rate=50.0,
                oos_cumulative_r=0.72, oos_max_drawdown_r=0.5, oos_payoff_ratio=1.3,
                expectancy_degradation=0.6, pf_degradation=0.8, wr_degradation=0.91,
            ),
            FoldResult(
                fold_index=2,
                candidate="strat_a",
                train_start="2024-04-01", train_end="2025-04-01",
                test_start="2025-04-01", test_end="2025-07-01",
                is_trades=18, is_expectancy_r=0.18, is_profit_factor=1.4, is_win_rate=52.0,
                is_cumulative_r=3.24, is_max_drawdown_r=1.5, is_payoff_ratio=1.6,
                oos_trades=5, oos_expectancy_r=0.08, oos_profit_factor=1.1, oos_win_rate=48.0,
                oos_cumulative_r=0.40, oos_max_drawdown_r=0.3, oos_payoff_ratio=1.1,
                expectancy_degradation=0.44, pf_degradation=0.79, wr_degradation=0.92,
            ),
        ]
        report = WalkForwardValidator._aggregate(
            candidate="strat_a",
            scheme="rolling",
            folds=folds,
        )
        assert report.num_folds == 2
        assert report.total_is_trades == 38
        assert report.total_oos_trades == 11
        assert report.oos_positive_folds == 2
        assert report.oos_negative_folds == 0
        assert report.overfitting_probability == 0.0
        assert report.cumulative_oos_r > 0
        assert report.avg_oos_expectancy_r > 0
        assert report.oos_sharpe_equivalent > 0

    def test_aggregate_mixed_folds_detects_overfitting(self) -> None:
        folds = [
            FoldResult(
                fold_index=1,
                candidate="strat_b",
                train_start="2024-01-01", train_end="2025-01-01",
                test_start="2025-01-01", test_end="2025-04-01",
                is_trades=30, is_expectancy_r=0.30, is_profit_factor=2.0, is_win_rate=60.0,
                is_cumulative_r=9.0, is_max_drawdown_r=1.5, is_payoff_ratio=2.0,
                oos_trades=8, oos_expectancy_r=0.05, oos_profit_factor=1.05, oos_win_rate=51.0,
                oos_cumulative_r=0.40, oos_max_drawdown_r=0.8, oos_payoff_ratio=1.1,
                expectancy_degradation=0.17, pf_degradation=0.525, wr_degradation=0.85,
            ),
            FoldResult(
                fold_index=2,
                candidate="strat_b",
                train_start="2024-04-01", train_end="2025-04-01",
                test_start="2025-04-01", test_end="2025-07-01",
                is_trades=25, is_expectancy_r=0.25, is_profit_factor=1.8, is_win_rate=58.0,
                is_cumulative_r=6.25, is_max_drawdown_r=2.0, is_payoff_ratio=1.9,
                oos_trades=7, oos_expectancy_r=-0.10, oos_profit_factor=0.8, oos_win_rate=42.0,
                oos_cumulative_r=-0.70, oos_max_drawdown_r=1.2, oos_payoff_ratio=0.9,
                expectancy_degradation=-0.4, pf_degradation=0.44, wr_degradation=0.72,
            ),
        ]
        report = WalkForwardValidator._aggregate(
            candidate="strat_b",
            scheme="rolling",
            folds=folds,
        )
        assert report.oos_positive_folds == 1
        assert report.oos_negative_folds == 1
        assert report.overfitting_probability == 50.0
        # High IS performance but degraded OOS → potential overfitting
        assert report.avg_is_expectancy_r > report.avg_oos_expectancy_r

    def test_aggregate_empty_folds(self) -> None:
        report = WalkForwardValidator._aggregate(
            candidate="empty",
            scheme="rolling",
            folds=[],
        )
        assert report.num_folds == 0
        assert report.total_is_trades == 0
        assert report.total_oos_trades == 0
