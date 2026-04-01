from __future__ import annotations

from scripts.run_exhaustion_candidate_vs_challenger_managed import build_comparison_decision


def _summary_row(
    *,
    cost: str,
    window: str,
    kind: str,
    geo: float,
    pf: float,
    max_dd_r: float,
) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "profile_kind": kind,
        "geometric_return_pct": geo,
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
    }


def _side_row(*, cost: str, window: str, kind: str, side: str, cum_r: float) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "profile_kind": kind,
        "side": side,
        "cum_r": cum_r,
    }


def test_build_comparison_decision_candidate_can_challenge_mainline() -> None:
    summary_rows = [
        _summary_row(cost="base", window="full_2020", kind="baseline_managed", geo=30.0, pf=1.20, max_dd_r=13.0),
        _summary_row(cost="base", window="full_2020", kind="candidate", geo=32.0, pf=1.25, max_dd_r=14.0),
        _summary_row(cost="stress_x2", window="full_2020", kind="baseline_managed", geo=5.0, pf=1.05, max_dd_r=15.0),
        _summary_row(cost="stress_x2", window="full_2020", kind="candidate", geo=6.0, pf=1.10, max_dd_r=14.0),
    ]
    side_rows = [
        _side_row(cost="base", window="two_year", kind="baseline_managed", side="LONG", cum_r=10.0),
        _side_row(cost="base", window="two_year", kind="candidate", side="LONG", cum_r=8.5),
        _side_row(cost="base", window="full_2020", kind="baseline_managed", side="SHORT", cum_r=6.0),
        _side_row(cost="base", window="full_2020", kind="candidate", side="SHORT", cum_r=9.0),
    ]

    decision = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        baseline_strategy_profile="baseline_profile",
        baseline_overlay_profile="overlay",
        candidate_profile="candidate_profile",
    )

    assert decision["status"] == "candidate_can_challenge_mainline"
    assert decision["pass_head_to_head_geo"] is True
    assert decision["pass_secondary_long_guard"] is True
    assert decision["pool_status"] == "independent_pool_member"


def test_build_comparison_decision_mainline_still_preferred_when_candidate_fails_guard() -> None:
    summary_rows = [
        _summary_row(cost="base", window="full_2020", kind="baseline_managed", geo=30.0, pf=1.20, max_dd_r=13.0),
        _summary_row(cost="base", window="full_2020", kind="candidate", geo=31.0, pf=1.25, max_dd_r=14.0),
        _summary_row(cost="stress_x2", window="full_2020", kind="baseline_managed", geo=5.0, pf=1.05, max_dd_r=15.0),
        _summary_row(cost="stress_x2", window="full_2020", kind="candidate", geo=-1.0, pf=0.95, max_dd_r=14.0),
    ]
    side_rows = [
        _side_row(cost="base", window="two_year", kind="baseline_managed", side="LONG", cum_r=10.0),
        _side_row(cost="base", window="two_year", kind="candidate", side="LONG", cum_r=7.0),
        _side_row(cost="base", window="full_2020", kind="baseline_managed", side="SHORT", cum_r=6.0),
        _side_row(cost="base", window="full_2020", kind="candidate", side="SHORT", cum_r=9.0),
    ]

    decision = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        baseline_strategy_profile="baseline_profile",
        baseline_overlay_profile="overlay",
        candidate_profile="candidate_profile",
    )

    assert decision["status"] == "mainline_still_preferred_candidate_kept_in_pool"
    assert decision["pass_candidate_stress_geo"] is False
    assert decision["pass_candidate_stress_pf"] is False
