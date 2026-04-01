from __future__ import annotations

from scripts.run_switch_candidate_vs_ct_block80 import (
    CT_KIND,
    SWITCH_KIND,
    build_comparison_decision,
    build_cost_sensitivity,
)


def _summary_row(
    *,
    cost: str,
    window: str,
    kind: str,
    trades: int,
    geo: float,
    cagr: float,
    pf: float,
    max_dd_r: float,
) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "profile_kind": kind,
        "strategy_profile": kind,
        "profile_label": kind,
        "trades": trades,
        "geometric_return_pct": geo,
        "cagr_pct": cagr,
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
    }


def _side_row(*, cost: str, window: str, kind: str, side: str, cum_r: float) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "profile_kind": kind,
        "strategy_profile": kind,
        "profile_label": kind,
        "side": side,
        "cum_r": cum_r,
    }


def _concentration_row(*, kind: str, top3: float, best_year: float, best_month: float) -> dict:
    return {
        "profile_kind": kind,
        "strategy_profile": kind,
        "profile_label": kind,
        "top3_trade_pnl_share_pct": top3,
        "best_year_pnl_share_pct": best_year,
        "best_month_pnl_share_pct": best_month,
    }


def test_build_comparison_decision_promotes_switch_when_all_guards_pass() -> None:
    summary_rows = [
        _summary_row(
            cost="base",
            window="full_2020",
            kind=SWITCH_KIND,
            trades=140,
            geo=120.0,
            cagr=14.0,
            pf=1.35,
            max_dd_r=13.5,
        ),
        _summary_row(
            cost="base",
            window="full_2020",
            kind=CT_KIND,
            trades=40,
            geo=13.0,
            cagr=1.8,
            pf=2.10,
            max_dd_r=1.8,
        ),
        _summary_row(
            cost="stress_x2",
            window="full_2020",
            kind=SWITCH_KIND,
            trades=140,
            geo=95.0,
            cagr=12.0,
            pf=1.25,
            max_dd_r=13.6,
        ),
        _summary_row(
            cost="stress_x2",
            window="full_2020",
            kind=CT_KIND,
            trades=40,
            geo=11.0,
            cagr=1.6,
            pf=1.85,
            max_dd_r=1.9,
        ),
        _summary_row(
            cost="stress_x3",
            window="full_2020",
            kind=SWITCH_KIND,
            trades=140,
            geo=70.0,
            cagr=9.8,
            pf=1.12,
            max_dd_r=13.8,
        ),
        _summary_row(
            cost="stress_x3",
            window="full_2020",
            kind=CT_KIND,
            trades=40,
            geo=8.0,
            cagr=1.2,
            pf=1.40,
            max_dd_r=2.1,
        ),
    ]
    side_rows = [
        _side_row(cost="base", window="two_year", kind=SWITCH_KIND, side="LONG", cum_r=11.0),
        _side_row(cost="base", window="two_year", kind=CT_KIND, side="LONG", cum_r=1.0),
        _side_row(cost="base", window="full_2020", kind=SWITCH_KIND, side="SHORT", cum_r=20.0),
        _side_row(cost="base", window="full_2020", kind=CT_KIND, side="SHORT", cum_r=8.0),
    ]
    concentration_rows = [
        _concentration_row(kind=SWITCH_KIND, top3=18.0, best_year=32.0, best_month=17.0),
        _concentration_row(kind=CT_KIND, top3=45.0, best_year=50.0, best_month=24.0),
    ]

    result = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        switch_label="switch",
        ct_profile="ct",
    )

    assert result["status"] == "promoted_new_baseline_candidate"
    assert result["pass_head_to_head_geo"] is True
    assert result["pass_stress_x3_positive"] is True
    assert result["pass_top3_concentration"] is True


def test_build_comparison_decision_keeps_ct_when_switch_is_fragile() -> None:
    summary_rows = [
        _summary_row(
            cost="base",
            window="full_2020",
            kind=SWITCH_KIND,
            trades=12,
            geo=20.0,
            cagr=3.0,
            pf=0.95,
            max_dd_r=16.5,
        ),
        _summary_row(
            cost="base",
            window="full_2020",
            kind=CT_KIND,
            trades=40,
            geo=13.0,
            cagr=1.8,
            pf=2.10,
            max_dd_r=1.8,
        ),
        _summary_row(
            cost="stress_x2",
            window="full_2020",
            kind=SWITCH_KIND,
            trades=12,
            geo=1.0,
            cagr=0.2,
            pf=0.98,
            max_dd_r=16.6,
        ),
        _summary_row(
            cost="stress_x2",
            window="full_2020",
            kind=CT_KIND,
            trades=40,
            geo=11.0,
            cagr=1.6,
            pf=1.85,
            max_dd_r=1.9,
        ),
        _summary_row(
            cost="stress_x3",
            window="full_2020",
            kind=SWITCH_KIND,
            trades=12,
            geo=-2.0,
            cagr=-0.3,
            pf=0.90,
            max_dd_r=16.8,
        ),
        _summary_row(
            cost="stress_x3",
            window="full_2020",
            kind=CT_KIND,
            trades=40,
            geo=8.0,
            cagr=1.2,
            pf=1.40,
            max_dd_r=2.1,
        ),
    ]
    side_rows = [
        _side_row(cost="base", window="two_year", kind=SWITCH_KIND, side="LONG", cum_r=-3.0),
        _side_row(cost="base", window="two_year", kind=CT_KIND, side="LONG", cum_r=1.0),
    ]
    concentration_rows = [
        _concentration_row(kind=SWITCH_KIND, top3=72.0, best_year=91.0, best_month=55.0),
        _concentration_row(kind=CT_KIND, top3=45.0, best_year=50.0, best_month=24.0),
    ]

    result = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        switch_label="switch",
        ct_profile="ct",
    )

    assert result["status"] == "keep_ct_block80_as_independent_pool_member"
    assert result["pass_base_pf_guard"] is False
    assert result["pass_base_max_dd_guard"] is False
    assert result["pass_stress_x3_positive"] is False
    assert result["pass_top3_concentration"] is False


def test_build_cost_sensitivity_outputs_stress_x2_and_stress_x3_rows() -> None:
    summary_rows = [
        _summary_row(cost="base", window="full_2020", kind=SWITCH_KIND, trades=100, geo=50.0, cagr=7.0, pf=1.2, max_dd_r=10.0),
        _summary_row(cost="stress_x2", window="full_2020", kind=SWITCH_KIND, trades=100, geo=40.0, cagr=6.0, pf=1.1, max_dd_r=10.5),
        _summary_row(cost="stress_x3", window="full_2020", kind=SWITCH_KIND, trades=100, geo=30.0, cagr=5.0, pf=1.0, max_dd_r=11.0),
    ]

    rows = build_cost_sensitivity(summary_rows)
    stress_names = {row["stress_scenario"] for row in rows}

    assert stress_names == {"stress_x2", "stress_x3"}
