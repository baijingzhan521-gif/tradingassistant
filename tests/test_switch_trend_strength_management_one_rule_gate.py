from __future__ import annotations

import pytest

from scripts.run_switch_trend_strength_management_one_rule_gate import (
    BASELINE_PRESET,
    PRESETS,
    build_btc_yearly_comparison,
    build_oos_summary,
    build_promotion_decision,
)


def _summary_row(
    *,
    cost: str,
    window: str,
    preset: str,
    geo: float,
    cagr: float,
    pf: float,
    max_dd_r: float,
    avg_size: float,
) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "preset": preset,
        "geometric_return_pct": geo,
        "cagr_pct": cagr,
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
        "avg_size": avg_size,
    }


def _side_row(*, cost: str, window: str, preset: str, side: str, cum_r: float) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "preset": preset,
        "side": side,
        "cum_r": cum_r,
    }


def _concentration_row(*, cost: str, window: str, preset: str, top3: float, best_year: float) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "preset": preset,
        "top3_trades_pnl_share_pct": top3,
        "best_year_geometric_pct_share": best_year,
    }


def _build_inputs(
    *,
    candidate_overrides: dict[str, dict] | None = None,
    flat_override: dict | None = None,
    btc_counts: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, int]]:
    candidate_overrides = candidate_overrides or {}
    flat = {
        "base_geo": 10.0,
        "base_cagr": 18.0,
        "base_pf": 1.20,
        "base_dd": 5.0,
        "base_avg_size": 1.0,
        "stress_x2_geo": 3.0,
        "stress_x2_pf": 1.02,
        "stress_x3_geo": 1.0,
        "stress_x3_pf": 1.01,
        "two_year_long_cum_r": 20.0,
        "oos_geo": 1.0,
    }
    if flat_override:
        flat.update(flat_override)

    default_candidate = {
        "base_geo": 12.0,
        "base_cagr": 22.4,
        "base_pf": 1.16,
        "base_dd": 6.0,
        "base_avg_size": 1.0,
        "stress_x2_geo": 2.5,
        "stress_x2_pf": 1.01,
        "stress_x3_geo": 1.2,
        "stress_x3_pf": 1.00,
        "two_year_long_cum_r": 18.5,
        "top3": 55.0,
        "best_year": 70.0,
        "oos_geo": 2.0,
        "oos_available": True,
    }

    summary_rows: list[dict] = []
    side_rows: list[dict] = []
    concentration_rows: list[dict] = []
    oos_rows: list[dict] = []

    summary_rows.extend(
        [
            _summary_row(
                cost="base",
                window="full_2020",
                preset=BASELINE_PRESET,
                geo=flat["base_geo"],
                cagr=flat["base_cagr"],
                pf=flat["base_pf"],
                max_dd_r=flat["base_dd"],
                avg_size=flat["base_avg_size"],
            ),
            _summary_row(
                cost="stress_x2",
                window="full_2020",
                preset=BASELINE_PRESET,
                geo=flat["stress_x2_geo"],
                cagr=0.0,
                pf=flat["stress_x2_pf"],
                max_dd_r=flat["base_dd"],
                avg_size=flat["base_avg_size"],
            ),
            _summary_row(
                cost="stress_x3",
                window="full_2020",
                preset=BASELINE_PRESET,
                geo=flat["stress_x3_geo"],
                cagr=0.0,
                pf=flat["stress_x3_pf"],
                max_dd_r=flat["base_dd"],
                avg_size=flat["base_avg_size"],
            ),
        ]
    )
    side_rows.append(
        _side_row(cost="base", window="two_year", preset=BASELINE_PRESET, side="LONG", cum_r=flat["two_year_long_cum_r"])
    )
    oos_rows.append({"preset": BASELINE_PRESET, "geometric_return_pct": flat["oos_geo"], "oos_available": True})

    for preset in PRESETS:
        if preset == BASELINE_PRESET:
            continue
        values = default_candidate.copy()
        values.update(candidate_overrides.get(preset, {}))
        summary_rows.extend(
            [
                _summary_row(
                    cost="base",
                    window="full_2020",
                    preset=preset,
                    geo=values["base_geo"],
                    cagr=values["base_cagr"],
                    pf=values["base_pf"],
                    max_dd_r=values["base_dd"],
                    avg_size=values["base_avg_size"],
                ),
                _summary_row(
                    cost="stress_x2",
                    window="full_2020",
                    preset=preset,
                    geo=values["stress_x2_geo"],
                    cagr=0.0,
                    pf=values["stress_x2_pf"],
                    max_dd_r=values["base_dd"],
                    avg_size=values["base_avg_size"],
                ),
                _summary_row(
                    cost="stress_x3",
                    window="full_2020",
                    preset=preset,
                    geo=values["stress_x3_geo"],
                    cagr=0.0,
                    pf=values["stress_x3_pf"],
                    max_dd_r=values["base_dd"],
                    avg_size=values["base_avg_size"],
                ),
            ]
        )
        side_rows.append(
            _side_row(cost="base", window="two_year", preset=preset, side="LONG", cum_r=values["two_year_long_cum_r"])
        )
        concentration_rows.append(
            _concentration_row(
                cost="base",
                window="full_2020",
                preset=preset,
                top3=values["top3"],
                best_year=values["best_year"],
            )
        )
        oos_rows.append(
            {"preset": preset, "geometric_return_pct": values["oos_geo"], "oos_available": values["oos_available"]}
        )

    if btc_counts is None:
        btc_counts = {BASELINE_PRESET: 2, "long_trend95_k110": 2, "long_trend95_k115": 3, "long_trend95_k120": 1}
    return summary_rows, side_rows, concentration_rows, oos_rows, btc_counts


def test_build_promotion_decision_promotes_and_satisfies_btc_non_regression() -> None:
    winner = "long_trend95_k115"
    summary_rows, side_rows, concentration_rows, oos_rows, btc_counts = _build_inputs(
        candidate_overrides={
            winner: {"base_geo": 13.0, "base_cagr": 22.9, "oos_geo": 2.8},
            "long_trend95_k110": {"base_geo": 9.0, "base_cagr": 18.8},
            "long_trend95_k120": {"base_geo": 8.9, "base_cagr": 18.7},
        },
        btc_counts={BASELINE_PRESET: 2, "long_trend95_k110": 2, "long_trend95_k115": 3, "long_trend95_k120": 1},
    )
    decision, candidates = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary_rows=oos_rows,
        btc_outperform_counts=btc_counts,
    )

    assert decision["status"] == "promoted_management_overlay_candidate"
    assert decision["chosen_candidate_preset"] == winner
    assert decision["pass_btc_diagnostic_non_regression"] is True
    assert decision["satisfied_result"] is True
    assert any(row["preset"] == winner and row["all_pass"] for row in candidates)


@pytest.mark.parametrize(
    ("override", "expected_fail_key"),
    [
        ({"base_avg_size": 1.03}, "pass_budget_neutral"),
        ({"base_geo": 9.5}, "pass_base_geo"),
        ({"base_cagr": 17.8}, "pass_base_cagr"),
        ({"base_pf": 1.14}, "pass_base_pf"),
        ({"base_dd": 6.7}, "pass_base_max_dd"),
        ({"stress_x2_geo": 0.0}, "pass_stress_x2_geo"),
        ({"stress_x2_pf": 0.99}, "pass_stress_x2_pf"),
        ({"stress_x3_geo": 0.0}, "pass_stress_x3_geo"),
        ({"stress_x3_pf": 0.99}, "pass_stress_x3_pf"),
        ({"two_year_long_cum_r": 17.9}, "pass_long_guard"),
        ({"top3": 66.0}, "pass_top3_concentration"),
        ({"best_year": 81.0}, "pass_best_year_concentration"),
        ({"oos_geo": 0.5}, "pass_oos_geo_non_negative"),
        ({"oos_available": False}, "pass_oos_geo_non_negative"),
    ],
)
def test_build_promotion_decision_rejects_when_any_m1_gate_fails(override: dict, expected_fail_key: str) -> None:
    target = "long_trend95_k115"
    per_candidate = {
        preset: (override if preset == target else {"base_geo": 8.0, "base_cagr": 18.7})
        for preset in PRESETS
        if preset != BASELINE_PRESET
    }
    summary_rows, side_rows, concentration_rows, oos_rows, btc_counts = _build_inputs(candidate_overrides=per_candidate)
    decision, _ = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary_rows=oos_rows,
        btc_outperform_counts=btc_counts,
    )

    assert decision["chosen_candidate_preset"] == target
    assert decision["status"] == "rejected_management_overlay"
    assert decision[expected_fail_key] is False


def test_build_promotion_decision_keeps_status_promoted_even_if_btc_non_regression_fails() -> None:
    target = "long_trend95_k115"
    summary_rows, side_rows, concentration_rows, oos_rows, btc_counts = _build_inputs(
        candidate_overrides={
            target: {"base_geo": 13.0, "base_cagr": 22.9, "oos_geo": 2.8},
            "long_trend95_k110": {"base_geo": 9.0, "base_cagr": 18.8},
            "long_trend95_k120": {"base_geo": 8.9, "base_cagr": 18.7},
        },
        btc_counts={BASELINE_PRESET: 3, "long_trend95_k110": 2, "long_trend95_k115": 1, "long_trend95_k120": 1},
    )
    decision, _ = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary_rows=oos_rows,
        btc_outperform_counts=btc_counts,
    )

    assert decision["status"] == "promoted_management_overlay_candidate"
    assert decision["pass_btc_diagnostic_non_regression"] is False
    assert decision["satisfied_result"] is False


def test_build_promotion_decision_promoted_but_unsatisfied_when_cagr_target_missed() -> None:
    target = "long_trend95_k115"
    summary_rows, side_rows, concentration_rows, oos_rows, btc_counts = _build_inputs(
        candidate_overrides={
            target: {"base_geo": 13.0, "base_cagr": 21.95, "oos_geo": 2.8},
            "long_trend95_k110": {"base_geo": 9.0, "base_cagr": 18.8},
            "long_trend95_k120": {"base_geo": 8.9, "base_cagr": 18.7},
        },
        btc_counts={BASELINE_PRESET: 2, "long_trend95_k110": 2, "long_trend95_k115": 2, "long_trend95_k120": 1},
    )
    decision, _ = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary_rows=oos_rows,
        btc_outperform_counts=btc_counts,
    )

    assert decision["status"] == "promoted_management_overlay_candidate"
    assert decision["pass_target_cagr_22"] is False
    assert decision["satisfied_result"] is False


def test_build_oos_summary_returns_defaults_when_no_folds() -> None:
    rows = build_oos_summary([])
    assert len(rows) == len(PRESETS)
    assert {row["preset"] for row in rows} == set(PRESETS.keys())
    assert all(row["oos_available"] is False for row in rows)


def test_build_btc_yearly_comparison_aggregates_counts() -> None:
    yearly_rows = [
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "preset": BASELINE_PRESET,
            "year": 2020,
            "geometric_return_pct": 10.0,
        },
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "preset": BASELINE_PRESET,
            "year": 2021,
            "geometric_return_pct": 5.0,
        },
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "preset": "long_trend95_k110",
            "year": 2020,
            "geometric_return_pct": 12.0,
        },
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "preset": "long_trend95_k110",
            "year": 2021,
            "geometric_return_pct": 3.0,
        },
    ]
    btc = {2020: 8.0, 2021: 4.0}

    rows, counts = build_btc_yearly_comparison(yearly_rows=yearly_rows, btc_yearly_returns=btc)

    assert counts[BASELINE_PRESET] == 2
    assert counts["long_trend95_k110"] == 1
    assert any(row["row_type"] == "aggregate" and row["preset"] == BASELINE_PRESET for row in rows)
