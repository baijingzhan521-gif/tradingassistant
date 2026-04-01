from __future__ import annotations

from scripts.run_trend_divergence_standalone_one_rule_gate import (
    BASELINE_PROFILE,
    build_promotion_decision,
)


def _summary_row(
    *,
    cost: str,
    window: str,
    profile: str,
    geo: float,
    cagr: float,
    pf: float,
    max_dd_r: float,
    trades: int,
) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "strategy_profile": profile,
        "geometric_return_pct": geo,
        "cagr_pct": cagr,
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
        "trades": trades,
    }


def _concentration_row(*, cost: str, window: str, profile: str, top3_share: float, best_year_share: float) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "strategy_profile": profile,
        "top3_trades_pnl_share_pct": top3_share,
        "best_year_geometric_pct_share": best_year_share,
    }


def test_build_promotion_decision_promotes_candidate_when_all_gates_pass() -> None:
    candidate = "candidate_profile"
    summary_rows = [
        _summary_row(
            cost="base",
            window="full_2020",
            profile=BASELINE_PROFILE,
            geo=7.5,
            cagr=1.2,
            pf=1.08,
            max_dd_r=3.1,
            trades=13,
        ),
        _summary_row(
            cost="base",
            window="full_2020",
            profile=candidate,
            geo=12.2,
            cagr=2.0,
            pf=1.20,
            max_dd_r=4.0,
            trades=14,
        ),
        _summary_row(
            cost="stress_x2",
            window="full_2020",
            profile=candidate,
            geo=1.8,
            cagr=0.3,
            pf=1.04,
            max_dd_r=4.4,
            trades=14,
        ),
    ]
    concentration_rows = [
        _concentration_row(cost="base", window="full_2020", profile=candidate, top3_share=55.0, best_year_share=70.0)
    ]

    decision_rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )

    assert len(decision_rows) == 1
    decision = decision_rows[0]
    assert decision["candidate_profile"] == candidate
    assert decision["status"] == "promoted_standalone_candidate"
    assert decision["pass_base_geo_vs_baseline"] is True
    assert decision["pass_stress_pf"] is True
    assert decision["pass_best_year_share"] is True


def test_build_promotion_decision_rejects_when_profitability_gate_fails() -> None:
    candidate = "candidate_profile"
    summary_rows = [
        _summary_row(
            cost="base",
            window="full_2020",
            profile=BASELINE_PROFILE,
            geo=7.5,
            cagr=1.2,
            pf=1.08,
            max_dd_r=3.1,
            trades=13,
        ),
        _summary_row(
            cost="base",
            window="full_2020",
            profile=candidate,
            geo=12.2,
            cagr=2.0,
            pf=1.05,
            max_dd_r=4.0,
            trades=14,
        ),
        _summary_row(
            cost="stress_x2",
            window="full_2020",
            profile=candidate,
            geo=1.8,
            cagr=0.3,
            pf=1.04,
            max_dd_r=4.4,
            trades=14,
        ),
    ]
    concentration_rows = [
        _concentration_row(cost="base", window="full_2020", profile=candidate, top3_share=55.0, best_year_share=70.0)
    ]

    decision = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )[0]

    assert decision["status"] == "rejected_fragile_or_unprofitable"
    assert decision["pass_base_pf"] is False


def test_build_promotion_decision_rejects_when_concentration_gate_fails() -> None:
    candidate = "candidate_profile"
    summary_rows = [
        _summary_row(
            cost="base",
            window="full_2020",
            profile=BASELINE_PROFILE,
            geo=7.5,
            cagr=1.2,
            pf=1.08,
            max_dd_r=3.1,
            trades=13,
        ),
        _summary_row(
            cost="base",
            window="full_2020",
            profile=candidate,
            geo=12.2,
            cagr=2.0,
            pf=1.20,
            max_dd_r=4.0,
            trades=14,
        ),
        _summary_row(
            cost="stress_x2",
            window="full_2020",
            profile=candidate,
            geo=1.8,
            cagr=0.3,
            pf=1.04,
            max_dd_r=4.4,
            trades=14,
        ),
    ]
    concentration_rows = [
        _concentration_row(cost="base", window="full_2020", profile=candidate, top3_share=72.0, best_year_share=86.0)
    ]

    decision = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )[0]

    assert decision["status"] == "rejected_fragile_or_unprofitable"
    assert decision["pass_top3_share"] is False
    assert decision["pass_best_year_share"] is False
