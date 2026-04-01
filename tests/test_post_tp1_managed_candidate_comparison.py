from __future__ import annotations

from scripts.run_post_tp1_managed_candidate_comparison import (
    build_comparison_decision,
    build_offset_summary,
)


CHAMPION = "swing_trend_long_regime_gate_v1"
CHALLENGER = "swing_trend_long_regime_short_no_reversal_no_aux_v1"


def test_build_comparison_decision_prefers_challenger_when_fixed_guardrails_all_pass() -> None:
    summary_rows = [
        {
            "window": "full_2020",
            "baseline_strategy_profile": CHAMPION,
            "cum_r": 9.1,
            "profit_factor": 1.06,
            "max_dd_r": 21.2,
        },
        {
            "window": "full_2020",
            "baseline_strategy_profile": CHALLENGER,
            "cum_r": 27.2,
            "profit_factor": 1.20,
            "max_dd_r": 13.4,
        },
    ]
    side_rows = [
        {
            "window": "two_year",
            "baseline_strategy_profile": CHAMPION,
            "side": "LONG",
            "cum_r": 13.5,
        },
        {
            "window": "two_year",
            "baseline_strategy_profile": CHALLENGER,
            "side": "LONG",
            "cum_r": 13.5,
        },
        {
            "window": "full_2020",
            "baseline_strategy_profile": CHAMPION,
            "side": "LONG",
            "cum_r": 12.8,
        },
        {
            "window": "full_2020",
            "baseline_strategy_profile": CHALLENGER,
            "side": "LONG",
            "cum_r": 12.8,
        },
        {
            "window": "full_2020",
            "baseline_strategy_profile": CHAMPION,
            "side": "SHORT",
            "cum_r": -3.7,
        },
        {
            "window": "full_2020",
            "baseline_strategy_profile": CHALLENGER,
            "side": "SHORT",
            "cum_r": 14.4,
        },
    ]

    result = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        champion=CHAMPION,
        challenger=CHALLENGER,
    )

    assert result["status"] == "challenger_managed_preferred"
    assert result["primary_long_delta_r"] == 0.0
    assert result["primary_short_delta_r"] == 18.1


def test_build_offset_summary_counts_offset_months_by_window_and_scope() -> None:
    monthly_rows = [
        {
            "window": "full_2020",
            "scope": "overall",
            "month": "2022-10",
            "champion_cumulative_r": -0.2,
            "challenger_cumulative_r": 4.3,
            "delta_r": 4.5,
        },
        {
            "window": "full_2020",
            "scope": "overall",
            "month": "2025-12",
            "champion_cumulative_r": 0.1,
            "challenger_cumulative_r": -1.6,
            "delta_r": -1.7,
        },
        {
            "window": "full_2020",
            "scope": "overall",
            "month": "2026-01",
            "champion_cumulative_r": 0.3,
            "challenger_cumulative_r": 3.2,
            "delta_r": 2.9,
        },
    ]

    rows = build_offset_summary(monthly_rows)

    assert rows == [
        {
            "window": "full_2020",
            "scope": "overall",
            "champion_negative_challenger_positive_months": 1,
            "champion_negative_challenger_positive_delta_r": 4.5,
            "challenger_negative_champion_positive_months": 1,
            "challenger_negative_champion_positive_delta_r": -1.7,
        }
    ]
