from __future__ import annotations

from scripts.run_independent_candidate_vs_switch_baseline import build_comparison_decision


def _summary_row(
    *,
    cost: str,
    kind: str,
    geo: float,
    cagr: float,
    pf: float,
    dd: float,
) -> dict:
    return {
        "cost_scenario": cost,
        "window": "full_2020",
        "profile_kind": kind,
        "strategy_profile": kind,
        "geometric_return_pct": geo,
        "cagr_pct": cagr,
        "profit_factor": pf,
        "max_dd_r": dd,
    }


def _side_row(*, kind: str, long_cum_r: float) -> dict:
    return {
        "cost_scenario": "base",
        "window": "two_year",
        "profile_kind": kind,
        "strategy_profile": kind,
        "side": "LONG",
        "cum_r": long_cum_r,
    }


def _concentration_row(*, kind: str, top3: float, best_year: float) -> dict:
    return {
        "cost_scenario": "base",
        "window": "full_2020",
        "profile_kind": kind,
        "strategy_profile": kind,
        "top3_trades_pnl_share_pct": top3,
        "best_year_geometric_pct_share": best_year,
    }


def test_build_comparison_decision_candidate_beats_switch_when_all_pass() -> None:
    summary_rows = [
        _summary_row(cost="base", kind="switch_baseline", geo=80.0, cagr=8.0, pf=1.22, dd=10.0),
        _summary_row(cost="stress_x2", kind="switch_baseline", geo=40.0, cagr=4.0, pf=1.08, dd=11.0),
        _summary_row(cost="stress_x3", kind="switch_baseline", geo=20.0, cagr=2.0, pf=1.01, dd=12.0),
        _summary_row(cost="base", kind="candidate", geo=95.0, cagr=9.5, pf=1.20, dd=11.1),
        _summary_row(cost="stress_x2", kind="candidate", geo=25.0, cagr=2.7, pf=1.03, dd=11.8),
        _summary_row(cost="stress_x3", kind="candidate", geo=10.0, cagr=1.1, pf=1.00, dd=12.8),
    ]
    side_rows = [
        _side_row(kind="switch_baseline", long_cum_r=15.0),
        _side_row(kind="candidate", long_cum_r=13.1),
    ]
    concentration_rows = [
        _concentration_row(kind="switch_baseline", top3=20.0, best_year=40.0),
        _concentration_row(kind="candidate", top3=30.0, best_year=55.0),
    ]
    oos_summary = {
        "candidate_geometric_return_pct": 12.0,
        "baseline_geometric_return_pct": 8.0,
        "delta_vs_baseline_geometric_return_pct": 4.0,
        "oos_available": True,
    }

    result = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary=oos_summary,
        candidate_profile="trend_pullback_trend70_v1",
    )

    assert result["status"] == "candidate_beats_switch"
    assert result["pass_base_geo"] is True
    assert result["pass_oos_geo_non_negative"] is True


def test_build_comparison_decision_rejects_when_oos_and_geo_fail() -> None:
    summary_rows = [
        _summary_row(cost="base", kind="switch_baseline", geo=80.0, cagr=8.0, pf=1.22, dd=10.0),
        _summary_row(cost="stress_x2", kind="switch_baseline", geo=40.0, cagr=4.0, pf=1.08, dd=11.0),
        _summary_row(cost="stress_x3", kind="switch_baseline", geo=20.0, cagr=2.0, pf=1.01, dd=12.0),
        _summary_row(cost="base", kind="candidate", geo=70.0, cagr=7.0, pf=1.22, dd=10.8),
        _summary_row(cost="stress_x2", kind="candidate", geo=22.0, cagr=2.4, pf=1.01, dd=11.4),
        _summary_row(cost="stress_x3", kind="candidate", geo=9.0, cagr=1.0, pf=1.00, dd=12.5),
    ]
    side_rows = [
        _side_row(kind="switch_baseline", long_cum_r=15.0),
        _side_row(kind="candidate", long_cum_r=13.5),
    ]
    concentration_rows = [
        _concentration_row(kind="switch_baseline", top3=20.0, best_year=40.0),
        _concentration_row(kind="candidate", top3=30.0, best_year=55.0),
    ]
    oos_summary = {
        "candidate_geometric_return_pct": 5.0,
        "baseline_geometric_return_pct": 8.0,
        "delta_vs_baseline_geometric_return_pct": -3.0,
        "oos_available": True,
    }

    result = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary=oos_summary,
        candidate_profile="intraday_mtf_v2_trend70_v1",
    )

    assert result["status"] == "switch_baseline_retained"
    assert result["pass_base_geo"] is False
    assert result["pass_oos_geo_non_negative"] is False
