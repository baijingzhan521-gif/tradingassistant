from __future__ import annotations

from scripts.run_post_tp1_extension_matrix import (
    ROOT,
    build_acceptance_rows,
    classify_overlay_results,
    resolve_baseline_dir_map,
)


CHAMPION = "swing_trend_long_regime_gate_v1"
CHALLENGER = "swing_trend_long_regime_short_no_reversal_no_aux_v1"


def make_summary_row(
    *,
    baseline_strategy_profile: str,
    window: str,
    profile: str,
    cum_r: float,
    pf: float,
    max_dd_r: float,
) -> dict[str, object]:
    return {
        "window": window,
        "baseline_strategy_profile": baseline_strategy_profile,
        "baseline_profile_label": baseline_strategy_profile,
        "baseline_source": "direct_backtest",
        "profile": profile,
        "label": profile,
        "cum_r": cum_r,
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
    }


def make_side_row(
    *,
    baseline_strategy_profile: str,
    window: str,
    profile: str,
    side: str,
    cum_r: float,
) -> dict[str, object]:
    return {
        "window": window,
        "baseline_strategy_profile": baseline_strategy_profile,
        "baseline_profile_label": baseline_strategy_profile,
        "baseline_source": "direct_backtest",
        "profile": profile,
        "label": profile,
        "side": side,
        "cum_r": cum_r,
    }


def test_resolve_baseline_dir_map_applies_legacy_dir_to_mainline_only_when_multiple_profiles() -> None:
    result = resolve_baseline_dir_map(
        legacy_baseline_dir="artifacts/backtests/stop_ablation_mainline",
        explicit_mapping_text=None,
        strategy_profiles=[CHAMPION, CHALLENGER],
    )

    assert result == {CHAMPION: ROOT / "artifacts/backtests/stop_ablation_mainline"}


def test_build_acceptance_rows_evaluates_each_baseline_independently() -> None:
    summary_rows = [
        make_summary_row(
            baseline_strategy_profile=CHAMPION,
            window="full_2020",
            profile="baseline_be_after_tp1",
            cum_r=3.4,
            pf=1.02,
            max_dd_r=26.0,
        ),
        make_summary_row(
            baseline_strategy_profile=CHAMPION,
            window="full_2020",
            profile="be_if_no_extension_within_3bars_after_tp1",
            cum_r=7.8,
            pf=1.06,
            max_dd_r=21.7,
        ),
        make_summary_row(
            baseline_strategy_profile=CHALLENGER,
            window="full_2020",
            profile="baseline_be_after_tp1",
            cum_r=24.0,
            pf=1.20,
            max_dd_r=15.5,
        ),
        make_summary_row(
            baseline_strategy_profile=CHALLENGER,
            window="full_2020",
            profile="be_if_no_extension_within_3bars_after_tp1",
            cum_r=25.0,
            pf=1.22,
            max_dd_r=14.0,
        ),
    ]
    side_rows = [
        make_side_row(
            baseline_strategy_profile=CHAMPION,
            window="two_year",
            profile="baseline_be_after_tp1",
            side="LONG",
            cum_r=14.7,
        ),
        make_side_row(
            baseline_strategy_profile=CHAMPION,
            window="two_year",
            profile="be_if_no_extension_within_3bars_after_tp1",
            side="LONG",
            cum_r=11.4,
        ),
        make_side_row(
            baseline_strategy_profile=CHALLENGER,
            window="two_year",
            profile="baseline_be_after_tp1",
            side="LONG",
            cum_r=14.7,
        ),
        make_side_row(
            baseline_strategy_profile=CHALLENGER,
            window="two_year",
            profile="be_if_no_extension_within_3bars_after_tp1",
            side="LONG",
            cum_r=13.5,
        ),
    ]

    rows = build_acceptance_rows(
        summary_rows=summary_rows,
        side_rows=side_rows,
        baseline_strategy_profiles=[CHAMPION, CHALLENGER],
        overlay_profiles=["be_if_no_extension_within_3bars_after_tp1"],
        acceptance_window="full_2020",
        secondary_window="two_year",
    )

    by_baseline = {row["baseline_strategy_profile"]: row for row in rows}
    assert by_baseline[CHAMPION]["qualified"] is False
    assert by_baseline[CHAMPION]["pass_secondary_long_guard"] is False
    assert by_baseline[CHALLENGER]["qualified"] is True
    assert by_baseline[CHALLENGER]["pass_secondary_long_guard"] is True


def test_classify_overlay_results_returns_universal_overlay_when_one_profile_qualifies_everywhere() -> None:
    acceptance_rows = [
        {
            "baseline_strategy_profile": CHAMPION,
            "overlay_profile": "be_if_no_extension_within_3bars_after_tp1",
            "overlay_label": "3 Bars",
            "delta_cum_r": 3.0,
            "delta_profit_factor": 0.03,
            "delta_max_dd_r": -4.0,
            "qualified": True,
        },
        {
            "baseline_strategy_profile": CHALLENGER,
            "overlay_profile": "be_if_no_extension_within_3bars_after_tp1",
            "overlay_label": "3 Bars",
            "delta_cum_r": 2.0,
            "delta_profit_factor": 0.02,
            "delta_max_dd_r": -1.0,
            "qualified": True,
        },
    ]

    result = classify_overlay_results(
        acceptance_rows=acceptance_rows,
        baseline_strategy_profiles=[CHAMPION, CHALLENGER],
        champion_profile=CHAMPION,
        challenger_profile=CHALLENGER,
    )

    assert result["classification"] == "universal_overlay"
    assert result["selected_overlay_profile"] == "be_if_no_extension_within_3bars_after_tp1"


def test_classify_overlay_results_returns_challenger_only_overlay() -> None:
    acceptance_rows = [
        {
            "baseline_strategy_profile": CHAMPION,
            "overlay_profile": "be_if_no_extension_within_3bars_after_tp1",
            "overlay_label": "3 Bars",
            "delta_cum_r": 3.0,
            "delta_profit_factor": 0.03,
            "delta_max_dd_r": -4.0,
            "qualified": False,
        },
        {
            "baseline_strategy_profile": CHALLENGER,
            "overlay_profile": "be_if_no_extension_within_3bars_after_tp1",
            "overlay_label": "3 Bars",
            "delta_cum_r": 2.0,
            "delta_profit_factor": 0.02,
            "delta_max_dd_r": -1.0,
            "qualified": True,
        },
    ]

    result = classify_overlay_results(
        acceptance_rows=acceptance_rows,
        baseline_strategy_profiles=[CHAMPION, CHALLENGER],
        champion_profile=CHAMPION,
        challenger_profile=CHALLENGER,
    )

    assert result["classification"] == "challenger_only_overlay"
    assert result["selected_overlay_profile"] == "be_if_no_extension_within_3bars_after_tp1"


def test_classify_overlay_results_rejects_champion_only_gain() -> None:
    acceptance_rows = [
        {
            "baseline_strategy_profile": CHAMPION,
            "overlay_profile": "be_if_no_extension_within_3bars_after_tp1",
            "overlay_label": "3 Bars",
            "delta_cum_r": 3.0,
            "delta_profit_factor": 0.03,
            "delta_max_dd_r": -4.0,
            "qualified": True,
        },
        {
            "baseline_strategy_profile": CHALLENGER,
            "overlay_profile": "be_if_no_extension_within_3bars_after_tp1",
            "overlay_label": "3 Bars",
            "delta_cum_r": -1.0,
            "delta_profit_factor": -0.01,
            "delta_max_dd_r": 1.0,
            "qualified": False,
        },
    ]

    result = classify_overlay_results(
        acceptance_rows=acceptance_rows,
        baseline_strategy_profiles=[CHAMPION, CHALLENGER],
        champion_profile=CHAMPION,
        challenger_profile=CHALLENGER,
    )

    assert result["classification"] == "rejected"
    assert result["selected_overlay_profile"] is None
