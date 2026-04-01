from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import scripts.run_range_failure_vs_challenger_managed as module


BASELINE = "swing_trend_long_regime_short_no_reversal_no_aux_v1"
OVERLAY = "be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98"
ALT = "swing_range_failure_v1_btc"


def make_summary_row(
    *,
    cost_scenario: str,
    window: str,
    profile_kind: str,
    cum_r: float,
    pf: float,
    max_dd_r: float,
) -> dict[str, object]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "profile_kind": profile_kind,
        "cum_r": cum_r,
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
    }


def make_complementarity_row(
    *,
    cost_scenario: str,
    window: str,
    monthly_corr: float | None,
    offset_months: int,
    offset_r_sum: float,
    opposite_sign_months: int = 0,
) -> dict[str, object]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "monthly_corr": monthly_corr,
        "baseline_negative_alt_positive_months": offset_months,
        "offset_r_sum": offset_r_sum,
        "opposite_sign_months": opposite_sign_months,
    }


def test_run_managed_baseline_with_helper_uses_replay_helper(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeStrategy:
        window_config = {"trigger_timeframe": "1h"}

    class FakeStrategyService:
        def build_strategy(self, strategy_profile: str) -> FakeStrategy:
            captured["strategy_profile"] = strategy_profile
            return FakeStrategy()

    class FakeService:
        strategy_service = FakeStrategyService()

    def fake_run_profile(**kwargs):
        captured["overlay_spec"] = kwargs["spec"].name
        captured["start"] = kwargs["start"]
        captured["end"] = kwargs["end"]
        return "summary", ["trade"], []

    monkeypatch.setattr(module, "run_profile", fake_run_profile)

    summary, trades = module.run_managed_baseline_with_helper(
        service=FakeService(),
        symbol="BTC/USDT:USDT",
        baseline_strategy_profile=BASELINE,
        baseline_overlay_profile=OVERLAY,
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 2, 1, tzinfo=timezone.utc),
        enriched_frames={
            "1h": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
                    "high": [100.0],
                    "close": [99.0],
                }
            )
        },
    )

    assert summary == "summary"
    assert trades == ["trade"]
    assert captured["strategy_profile"] == BASELINE
    assert captured["overlay_spec"] == OVERLAY


def test_build_decision_returns_rejected_floor_when_standalone_floor_fails() -> None:
    summary_rows = [
        make_summary_row(cost_scenario="base", window="full_2020", profile_kind="baseline_managed", cum_r=30.0, pf=1.21, max_dd_r=13.4),
        make_summary_row(cost_scenario="base", window="full_2020", profile_kind="alt", cum_r=-12.0, pf=0.82, max_dd_r=19.0),
        make_summary_row(cost_scenario="base", window="two_year", profile_kind="alt", cum_r=-1.0, pf=0.91, max_dd_r=8.0),
        make_summary_row(cost_scenario="stress_x2", window="full_2020", profile_kind="alt", cum_r=-9.0, pf=0.79, max_dd_r=19.0),
    ]
    complementarity_rows = [
        make_complementarity_row(cost_scenario="base", window="full_2020", monthly_corr=-0.2, offset_months=4, offset_r_sum=5.0),
        make_complementarity_row(cost_scenario="base", window="two_year", monthly_corr=-0.1, offset_months=1, offset_r_sum=1.0),
    ]

    result = module.build_decision(
        summary_rows=summary_rows,
        complementarity_rows=complementarity_rows,
        baseline_strategy_profile=BASELINE,
        baseline_overlay_profile=OVERLAY,
        alt_profile=ALT,
    )

    assert result["status"] == "rejected_floor"
    assert result["pass_base_full_2020_cum_r_floor"] is False


def test_build_decision_returns_rejected_offset_when_floor_passes_but_complementarity_fails() -> None:
    summary_rows = [
        make_summary_row(cost_scenario="base", window="full_2020", profile_kind="baseline_managed", cum_r=30.0, pf=1.21, max_dd_r=13.4),
        make_summary_row(cost_scenario="base", window="full_2020", profile_kind="alt", cum_r=-2.0, pf=0.91, max_dd_r=11.0),
        make_summary_row(cost_scenario="base", window="two_year", profile_kind="alt", cum_r=1.0, pf=0.95, max_dd_r=7.0),
        make_summary_row(cost_scenario="stress_x2", window="full_2020", profile_kind="alt", cum_r=-5.0, pf=0.78, max_dd_r=12.0),
    ]
    complementarity_rows = [
        make_complementarity_row(cost_scenario="base", window="full_2020", monthly_corr=0.35, offset_months=1, offset_r_sum=0.8),
        make_complementarity_row(cost_scenario="base", window="two_year", monthly_corr=0.10, offset_months=1, offset_r_sum=0.8),
    ]

    result = module.build_decision(
        summary_rows=summary_rows,
        complementarity_rows=complementarity_rows,
        baseline_strategy_profile=BASELINE,
        baseline_overlay_profile=OVERLAY,
        alt_profile=ALT,
    )

    assert result["status"] == "rejected_offset"
    assert result["pass_base_full_2020_pf_floor"] is True
    assert result["pass_full_2020_monthly_corr_gate"] is False


def test_build_decision_returns_watchlist_when_floor_and_offset_both_pass() -> None:
    summary_rows = [
        make_summary_row(cost_scenario="base", window="full_2020", profile_kind="baseline_managed", cum_r=30.0, pf=1.21, max_dd_r=13.4),
        make_summary_row(cost_scenario="base", window="full_2020", profile_kind="alt", cum_r=-3.0, pf=0.92, max_dd_r=12.0),
        make_summary_row(cost_scenario="base", window="two_year", profile_kind="alt", cum_r=1.5, pf=0.97, max_dd_r=7.0),
        make_summary_row(cost_scenario="stress_x2", window="full_2020", profile_kind="alt", cum_r=-8.0, pf=0.77, max_dd_r=13.0),
    ]
    complementarity_rows = [
        make_complementarity_row(cost_scenario="base", window="full_2020", monthly_corr=-0.12, offset_months=4, offset_r_sum=4.2),
        make_complementarity_row(cost_scenario="base", window="two_year", monthly_corr=-0.05, offset_months=2, offset_r_sum=1.3),
    ]

    result = module.build_decision(
        summary_rows=summary_rows,
        complementarity_rows=complementarity_rows,
        baseline_strategy_profile=BASELINE,
        baseline_overlay_profile=OVERLAY,
        alt_profile=ALT,
    )

    assert result["status"] == "complementary_watchlist"
    assert result["worth_continuing"] is True
    assert result["next_route"] == "hold_range_failure_watchlist"


def test_summarize_complementarity_aggregates_signal_month_metrics() -> None:
    baseline_trades = pd.DataFrame(
        {
            "signal_month": ["2025-01", "2025-02", "2025-03"],
            "pnl_r": [-1.0, 2.0, -1.0],
        }
    )
    alt_trades = pd.DataFrame(
        {
            "signal_month": ["2025-01", "2025-02", "2025-03"],
            "pnl_r": [1.0, -2.0, 1.0],
        }
    )

    summary, monthly_rows = module.summarize_complementarity(
        cost_scenario="base",
        window="full_2020",
        window_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2025, 4, 1, tzinfo=timezone.utc),
        baseline_trades=baseline_trades,
        alt_trades=alt_trades,
    )

    assert summary["monthly_corr"] == -1.0
    assert summary["baseline_negative_alt_positive_months"] == 2
    assert summary["offset_r_sum"] == 2.0
    assert len(monthly_rows) == 3
