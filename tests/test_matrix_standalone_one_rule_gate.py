from __future__ import annotations

from scripts.run_matrix_standalone_one_rule_gate import (
    BASELINE_PROFILE,
    build_promotion_decision,
    select_top_promoted_candidate,
    _strategy_behavior_signature,
    _validate_profile_pool,
)


def _summary_row(*, cost: str, profile: str, geo: float, cagr: float, pf: float, dd: float, trades: int) -> dict:
    return {
        "cost_scenario": cost,
        "window": "full_2020",
        "strategy_profile": profile,
        "geometric_return_pct": geo,
        "cagr_pct": cagr,
        "profit_factor": pf,
        "max_dd_r": dd,
        "trades": trades,
    }


def _concentration_row(*, profile: str, top3: float, best_year: float) -> dict:
    return {
        "cost_scenario": "base",
        "window": "full_2020",
        "strategy_profile": profile,
        "top3_trades_pnl_share_pct": top3,
        "best_year_geometric_pct_share": best_year,
    }


def test_matrix_profiles_are_strict_one_rule_by_behavior_signature() -> None:
    profiles = [
        BASELINE_PROFILE,
        "swing_trend_long_regime_gate_v1",
        "swing_trend_matrix_no_gate_simple_entry_v1",
    ]
    _validate_profile_pool(profiles)

    base_gate, base_current_entry = _strategy_behavior_signature(BASELINE_PROFILE)
    gate_candidate = _strategy_behavior_signature("swing_trend_long_regime_gate_v1")
    entry_candidate = _strategy_behavior_signature("swing_trend_matrix_no_gate_simple_entry_v1")
    assert base_gate is False
    assert base_current_entry is True
    assert gate_candidate == (True, True)
    assert entry_candidate == (False, False)


def test_build_promotion_decision_marks_rejected_inactive_candidate() -> None:
    candidate = "swing_trend_long_regime_gate_v1"
    summary_rows = [
        _summary_row(cost="base", profile=BASELINE_PROFILE, geo=10.0, cagr=2.0, pf=1.05, dd=7.0, trades=200),
        _summary_row(cost="stress_x2", profile=BASELINE_PROFILE, geo=6.0, cagr=1.2, pf=1.02, dd=8.0, trades=200),
        _summary_row(cost="stress_x3", profile=BASELINE_PROFILE, geo=2.0, cagr=0.4, pf=1.00, dd=9.0, trades=200),
        _summary_row(cost="base", profile=candidate, geo=12.0, cagr=2.4, pf=1.12, dd=5.8, trades=120),
        _summary_row(cost="stress_x2", profile=candidate, geo=3.0, cagr=0.8, pf=1.01, dd=6.2, trades=120),
        _summary_row(cost="stress_x3", profile=candidate, geo=1.0, cagr=0.3, pf=1.00, dd=6.5, trades=120),
    ]
    concentration_rows = [_concentration_row(profile=candidate, top3=42.0, best_year=60.0)]
    activation_rows = [
        {
            "candidate_profile": candidate,
            "pass_activation_precheck": False,
        }
    ]

    rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        activation_rows=activation_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )

    assert rows[0]["pass_activation_precheck"] is False
    assert rows[0]["status"] == "rejected_inactive_candidate"


def test_build_promotion_decision_promotes_when_activation_and_m1_all_pass() -> None:
    candidate = "swing_trend_matrix_no_gate_simple_entry_v1"
    summary_rows = [
        _summary_row(cost="base", profile=BASELINE_PROFILE, geo=10.0, cagr=2.0, pf=1.05, dd=7.0, trades=200),
        _summary_row(cost="stress_x2", profile=BASELINE_PROFILE, geo=6.0, cagr=1.2, pf=1.02, dd=8.0, trades=200),
        _summary_row(cost="stress_x3", profile=BASELINE_PROFILE, geo=2.0, cagr=0.4, pf=1.00, dd=9.0, trades=200),
        _summary_row(cost="base", profile=candidate, geo=12.0, cagr=2.4, pf=1.12, dd=5.8, trades=120),
        _summary_row(cost="stress_x2", profile=candidate, geo=3.0, cagr=0.8, pf=1.01, dd=6.2, trades=120),
        _summary_row(cost="stress_x3", profile=candidate, geo=1.0, cagr=0.3, pf=1.00, dd=6.5, trades=120),
    ]
    concentration_rows = [_concentration_row(profile=candidate, top3=42.0, best_year=60.0)]
    activation_rows = [
        {
            "candidate_profile": candidate,
            "pass_activation_precheck": True,
        }
    ]

    rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        activation_rows=activation_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )

    assert rows[0]["pass_activation_precheck"] is True
    assert rows[0]["status"] == "promoted_standalone_candidate"


def test_select_top_promoted_candidate_uses_geo_then_pf_then_dd() -> None:
    rows = [
        {
            "candidate_profile": "a",
            "status": "promoted_standalone_candidate",
            "base_full_2020_candidate_geometric_return_pct": 20.0,
            "base_full_2020_candidate_profit_factor": 1.20,
            "base_full_2020_candidate_max_dd_r": 6.0,
        },
        {
            "candidate_profile": "b",
            "status": "promoted_standalone_candidate",
            "base_full_2020_candidate_geometric_return_pct": 20.0,
            "base_full_2020_candidate_profit_factor": 1.20,
            "base_full_2020_candidate_max_dd_r": 5.0,
        },
        {
            "candidate_profile": "c",
            "status": "promoted_standalone_candidate",
            "base_full_2020_candidate_geometric_return_pct": 20.0,
            "base_full_2020_candidate_profit_factor": 1.22,
            "base_full_2020_candidate_max_dd_r": 5.5,
        },
    ]
    assert select_top_promoted_candidate(rows) == "c"
