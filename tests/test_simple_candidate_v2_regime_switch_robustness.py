from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import scripts.run_simple_candidate_v2_regime_switch_robustness as module


def test_build_probe_panel_row_uses_stress_outputs_and_long_guard() -> None:
    case = {
        "summary_rows": [
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "always_challenger_managed",
                "geometric_return_pct": 10.0,
                "profit_factor": 1.2,
                "max_dd_r": 5.0,
            },
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "geometric_return_pct": 15.0,
                "profit_factor": 1.3,
                "max_dd_r": 4.0,
            },
            {
                "cost_scenario": "stress_x2",
                "window": "full_2020",
                "scenario_kind": "always_challenger_managed",
                "geometric_return_pct": 5.0,
                "profit_factor": 1.0,
                "max_dd_r": 5.0,
            },
            {
                "cost_scenario": "stress_x2",
                "window": "full_2020",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "geometric_return_pct": 8.0,
                "profit_factor": 1.1,
                "max_dd_r": 4.0,
            },
            {
                "cost_scenario": "stress_x3",
                "window": "full_2020",
                "scenario_kind": "always_challenger_managed",
                "geometric_return_pct": 1.0,
                "profit_factor": 0.9,
                "max_dd_r": 5.0,
            },
            {
                "cost_scenario": "stress_x3",
                "window": "full_2020",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "geometric_return_pct": 2.0,
                "profit_factor": 1.0,
                "max_dd_r": 4.0,
            },
        ],
        "side_rows": [
            {
                "cost_scenario": "base",
                "window": "two_year",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "side": "LONG",
                "cum_r": 12.0,
            }
        ],
        "decision_row": {
            "status": "promoted_calendar_switch_candidate",
            "pass_base_geo": True,
            "pass_base_pf": True,
            "pass_base_max_dd": True,
            "pass_stress_geo": True,
            "pass_stress_pf": True,
        },
    }

    row = module.build_probe_panel_row(switch_date="2024-03-19", case=case)

    assert row["base_delta_geometric_return_pct"] == 5.0
    assert row["stress_x2_delta_geometric_return_pct"] == 3.0
    assert row["stress_x3_delta_geometric_return_pct"] == 1.0
    assert row["two_year_long_cum_r"] == 12.0
    assert row["stress_x3_beats_baseline"] is True


def test_build_concentration_rows_measures_trade_and_year_concentration() -> None:
    trade_frame = pd.DataFrame(
        [
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "signal_time": pd.Timestamp("2020-01-01T00:00:00Z"),
                "pnl_r": 3.0,
            },
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "signal_time": pd.Timestamp("2020-01-02T00:00:00Z"),
                "pnl_r": 2.0,
            },
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "signal_time": pd.Timestamp("2021-01-01T00:00:00Z"),
                "pnl_r": 1.0,
            },
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "signal_time": pd.Timestamp("2021-02-01T00:00:00Z"),
                "pnl_r": 1.0,
            },
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "signal_time": pd.Timestamp("2021-03-01T00:00:00Z"),
                "pnl_r": 1.0,
            },
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "always_challenger_managed",
                "signal_time": pd.Timestamp("2020-01-01T00:00:00Z"),
                "pnl_r": 1.0,
            },
        ]
    )
    case = {"trade_frame": trade_frame}

    rows = module.build_concentration_rows(case=case)
    switch_row = next(row for row in rows if row["scenario_kind"] == "switch_simple_candidate_v2_then_challenger_managed")

    assert switch_row["trades"] == 5
    assert switch_row["cum_r"] == 8.0
    assert switch_row["top3_trade_pnl_share_pct"] == 75.0
    assert switch_row["best_year_pnl_share_pct"] == 62.5
    assert switch_row["positive_years"] == 2


def test_build_robustness_decision_covers_all_branches() -> None:
    probe_rows = [
        {
            "switch_date": "2024-03-19",
            "base_pass": True,
            "stress_x2_pass": True,
            "stress_x3_positive": True,
            "stress_x3_beats_baseline": True,
            "stress_x3_pf_ge_baseline": True,
        },
        {
            "switch_date": "2024-04-18",
            "base_pass": True,
            "stress_x2_pass": True,
            "stress_x3_positive": True,
            "stress_x3_beats_baseline": True,
            "stress_x3_pf_ge_baseline": True,
        },
    ]
    concentration_rows = [
        {
            "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
            "top3_trade_pnl_share_pct": 30.0,
            "best_year_pnl_share_pct": 40.0,
            "best_month_pnl_share_pct": 20.0,
        }
    ]

    decision = module.build_robustness_decision(probe_rows=probe_rows, concentration_rows=concentration_rows)
    assert decision["status"] == "robust_under_probe_panel"

    concentrated = module.build_robustness_decision(
        probe_rows=probe_rows,
        concentration_rows=[
            {
                "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                "top3_trade_pnl_share_pct": 60.0,
                "best_year_pnl_share_pct": 70.0,
                "best_month_pnl_share_pct": 20.0,
            }
        ],
    )
    assert concentrated["status"] == "robust_but_concentrated"
    assert "concentration:top3_trade_share_gt_35pct" in concentrated["risk_flags"]

    fragile = module.build_robustness_decision(
        probe_rows=[{**probe_rows[0], "base_pass": False}],
        concentration_rows=concentration_rows,
    )
    assert fragile["status"] == "date_sensitive_or_cost_fragile"
