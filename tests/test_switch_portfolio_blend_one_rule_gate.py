from __future__ import annotations

import pytest

from scripts.run_switch_portfolio_blend_one_rule_gate import (
    BASELINE_PRESET,
    PRESETS,
    build_oos_selected_vs_flat,
    build_promotion_decision,
    validate_preset_weights,
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
) -> dict:
    return {
        "cost_scenario": cost,
        "window": window,
        "preset": preset,
        "geometric_return_pct": geo,
        "cagr_pct": cagr,
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
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
        "base_geo": 20.0,
        "base_cagr": 20.0,
        "base_pf": 1.20,
        "base_dd": 10.0,
        "stress_x2_geo": 8.0,
        "stress_x2_pf": 1.05,
        "stress_x3_geo": 3.0,
        "stress_x3_pf": 1.01,
        "two_year_long_cum_r": 18.0,
        "oos_geo": 1.0,
    }
    if flat_override:
        flat.update(flat_override)

    default_candidate = {
        "base_geo": 23.0,
        "base_cagr": 22.5,
        "base_pf": 1.16,
        "base_dd": 11.0,
        "stress_x2_geo": 4.0,
        "stress_x2_pf": 1.02,
        "stress_x3_geo": 1.2,
        "stress_x3_pf": 1.00,
        "two_year_long_cum_r": 16.5,
        "top3": 50.0,
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
            ),
            _summary_row(
                cost="stress_x2",
                window="full_2020",
                preset=BASELINE_PRESET,
                geo=flat["stress_x2_geo"],
                cagr=0.0,
                pf=flat["stress_x2_pf"],
                max_dd_r=flat["base_dd"],
            ),
            _summary_row(
                cost="stress_x3",
                window="full_2020",
                preset=BASELINE_PRESET,
                geo=flat["stress_x3_geo"],
                cagr=0.0,
                pf=flat["stress_x3_pf"],
                max_dd_r=flat["base_dd"],
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
                ),
                _summary_row(
                    cost="stress_x2",
                    window="full_2020",
                    preset=preset,
                    geo=values["stress_x2_geo"],
                    cagr=0.0,
                    pf=values["stress_x2_pf"],
                    max_dd_r=values["base_dd"],
                ),
                _summary_row(
                    cost="stress_x3",
                    window="full_2020",
                    preset=preset,
                    geo=values["stress_x3_geo"],
                    cagr=0.0,
                    pf=values["stress_x3_pf"],
                    max_dd_r=values["base_dd"],
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
        btc_counts = {
            BASELINE_PRESET: 2,
            "switch95_ct05": 3,
            "switch90_ct10": 1,
            "switch85_ct15": 1,
        }
    return summary_rows, side_rows, concentration_rows, oos_rows, btc_counts


def test_validate_preset_weights_rejects_invalid_sum() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        validate_preset_weights(
            {
                BASELINE_PRESET: {"switch_weight": 1.0, "ct_weight": 0.0},
                "bad": {"switch_weight": 0.9, "ct_weight": 0.2},
            }
        )


def test_build_promotion_decision_promotes_when_all_m1_and_satisfaction_pass() -> None:
    winner = "switch95_ct05"
    summary_rows, side_rows, concentration_rows, oos_rows, btc_counts = _build_inputs(
        candidate_overrides={
            winner: {"base_geo": 24.0, "base_cagr": 22.9, "oos_geo": 2.6},
            "switch90_ct10": {"base_geo": 19.0, "base_cagr": 19.5},
            "switch85_ct15": {"base_geo": 18.5, "base_cagr": 19.2},
        }
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
    assert decision["all_pass_m1"] is True
    assert decision["pass_target_cagr_22"] is True
    assert decision["pass_btc_diagnostic_non_regression"] is True
    assert any(row["preset"] == winner and row["all_pass"] for row in candidates)


@pytest.mark.parametrize(
    ("override", "expected_fail_key"),
    [
        ({"base_geo": 19.0}, "pass_base_geo"),
        ({"base_cagr": 19.8}, "pass_base_cagr"),
        ({"base_pf": 1.14}, "pass_base_pf"),
        ({"base_dd": 11.7}, "pass_base_max_dd"),
        ({"stress_x2_geo": 0.0}, "pass_stress_x2_geo"),
        ({"stress_x2_pf": 0.99}, "pass_stress_x2_pf"),
        ({"stress_x3_geo": 0.0}, "pass_stress_x3_geo"),
        ({"stress_x3_pf": 0.99}, "pass_stress_x3_pf"),
        ({"two_year_long_cum_r": 15.9}, "pass_long_guard"),
        ({"top3": 66.0}, "pass_top3_concentration"),
        ({"best_year": 81.0}, "pass_best_year_concentration"),
        ({"oos_geo": 0.8}, "pass_oos_geo_non_negative"),
        ({"oos_available": False}, "pass_oos_geo_non_negative"),
    ],
)
def test_build_promotion_decision_rejects_when_any_m1_gate_fails(override: dict, expected_fail_key: str) -> None:
    target = "switch95_ct05"
    per_candidate = {
        preset: (override if preset == target else {"base_geo": 18.0, "base_cagr": 18.0})
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


def test_build_promotion_decision_rejects_when_cagr_target_not_met() -> None:
    target = "switch95_ct05"
    summary_rows, side_rows, concentration_rows, oos_rows, btc_counts = _build_inputs(
        candidate_overrides={
            target: {"base_geo": 24.0, "base_cagr": 21.9, "oos_geo": 2.6},
            "switch90_ct10": {"base_geo": 18.9, "base_cagr": 18.9},
            "switch85_ct15": {"base_geo": 18.8, "base_cagr": 18.8},
        }
    )
    decision, _ = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary_rows=oos_rows,
        btc_outperform_counts=btc_counts,
    )

    assert decision["chosen_candidate_preset"] == target
    assert decision["all_pass_m1"] is True
    assert decision["pass_target_cagr_22"] is False
    assert decision["status"] == "rejected_management_overlay"


def test_build_promotion_decision_rejects_when_btc_non_regression_fails() -> None:
    target = "switch95_ct05"
    summary_rows, side_rows, concentration_rows, oos_rows, _ = _build_inputs(
        candidate_overrides={
            target: {"base_geo": 24.0, "base_cagr": 22.8, "oos_geo": 2.6},
            "switch90_ct10": {"base_geo": 18.9, "base_cagr": 18.9},
            "switch85_ct15": {"base_geo": 18.8, "base_cagr": 18.8},
        }
    )
    decision, _ = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary_rows=oos_rows,
        btc_outperform_counts={
            BASELINE_PRESET: 3,
            "switch95_ct05": 2,
            "switch90_ct10": 2,
            "switch85_ct15": 1,
        },
    )

    assert decision["chosen_candidate_preset"] == target
    assert decision["all_pass_m1"] is True
    assert decision["pass_btc_diagnostic_non_regression"] is False
    assert decision["status"] == "rejected_management_overlay"


def test_build_oos_selected_vs_flat_adds_aggregate_row() -> None:
    rows = build_oos_selected_vs_flat(
        fold_rows=[
            {
                "fold": 1,
                "preset": BASELINE_PRESET,
                "trades": 10,
                "geometric_return_pct": 1.0,
                "cagr_pct": 1.2,
                "profit_factor": 1.1,
                "cum_r": 2.0,
                "max_dd_r": 1.0,
            },
            {
                "fold": 1,
                "preset": "switch95_ct05",
                "trades": 10,
                "geometric_return_pct": 2.0,
                "cagr_pct": 2.1,
                "profit_factor": 1.2,
                "cum_r": 3.0,
                "max_dd_r": 1.1,
            },
            {
                "fold": 2,
                "preset": BASELINE_PRESET,
                "trades": 8,
                "geometric_return_pct": 3.0,
                "cagr_pct": 3.1,
                "profit_factor": 1.2,
                "cum_r": 4.0,
                "max_dd_r": 1.2,
            },
            {
                "fold": 2,
                "preset": "switch95_ct05",
                "trades": 8,
                "geometric_return_pct": 4.0,
                "cagr_pct": 4.2,
                "profit_factor": 1.3,
                "cum_r": 5.0,
                "max_dd_r": 1.3,
            },
        ],
        chosen_candidate_preset="switch95_ct05",
    )

    assert rows[-1]["fold"] == "aggregate"
    assert rows[-1]["candidate_trades"] == 18
    assert rows[-1]["flat_trades"] == 18
    assert float(rows[-1]["delta_geometric_return_pct"]) > 0.0
