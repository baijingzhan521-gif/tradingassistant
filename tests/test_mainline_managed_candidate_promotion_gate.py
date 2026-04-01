from __future__ import annotations

from scripts.post_tp1_managed_replay import build_service
from scripts.run_mainline_managed_candidate_promotion_gate import (
    PRIMARY_WINDOW,
    SECONDARY_WINDOW,
    build_concentration_summary,
    build_promotion_decision,
)


CHAMPION = "swing_trend_long_regime_gate_v1"
CHALLENGER = "swing_trend_long_regime_short_no_reversal_no_aux_v1"
OVERLAY = "be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98"


def make_summary_row(
    *,
    cost_scenario: str,
    window: str,
    strategy_profile: str,
    cum_r: float,
    pf: float,
    max_dd_r: float,
) -> dict[str, object]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "strategy_profile": strategy_profile,
        "cum_r": cum_r,
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
    }


def make_side_row(
    *,
    cost_scenario: str,
    window: str,
    strategy_profile: str,
    side: str,
    cum_r: float,
) -> dict[str, object]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "strategy_profile": strategy_profile,
        "side": side,
        "cum_r": cum_r,
    }


def test_build_service_applies_stress_cost_overrides() -> None:
    service = build_service(assumption_overrides={"taker_fee_bps": 10.0, "slippage_bps": 4.0})

    assert service.assumptions.taker_fee_bps == 10.0
    assert service.assumptions.slippage_bps == 4.0
    assert service.assumptions.scaled_tp2_r == 3.0


def test_build_promotion_decision_promotes_challenger_when_all_six_rules_pass() -> None:
    summary_rows = [
        make_summary_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHAMPION, cum_r=9.1, pf=1.06, max_dd_r=21.2),
        make_summary_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHALLENGER, cum_r=27.2, pf=1.20, max_dd_r=13.4),
        make_summary_row(cost_scenario="base", window=SECONDARY_WINDOW, strategy_profile=CHAMPION, cum_r=13.5, pf=1.07, max_dd_r=9.4),
        make_summary_row(cost_scenario="base", window=SECONDARY_WINDOW, strategy_profile=CHALLENGER, cum_r=13.5, pf=1.07, max_dd_r=9.4),
        make_summary_row(cost_scenario="stress_x2", window=PRIMARY_WINDOW, strategy_profile=CHAMPION, cum_r=5.8, pf=1.02, max_dd_r=22.8),
        make_summary_row(cost_scenario="stress_x2", window=PRIMARY_WINDOW, strategy_profile=CHALLENGER, cum_r=21.1, pf=1.09, max_dd_r=14.9),
    ]
    side_rows = [
        make_side_row(cost_scenario="base", window=SECONDARY_WINDOW, strategy_profile=CHAMPION, side="LONG", cum_r=13.5),
        make_side_row(cost_scenario="base", window=SECONDARY_WINDOW, strategy_profile=CHALLENGER, side="LONG", cum_r=13.5),
        make_side_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHAMPION, side="LONG", cum_r=12.8),
        make_side_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHALLENGER, side="LONG", cum_r=12.8),
        make_side_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHAMPION, side="SHORT", cum_r=-3.7),
        make_side_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHALLENGER, side="SHORT", cum_r=14.4),
    ]
    concentration_rows = [
        {"horizon": "quarter", "top3_positive_share_pct": 64.0, "regime_specialist_tendency": False},
        {"horizon": "rolling_180", "top3_positive_share_pct": 62.0, "regime_specialist_tendency": False},
        {"horizon": "rolling_365", "top3_positive_share_pct": 59.0, "regime_specialist_tendency": False},
    ]

    result = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        champion=CHAMPION,
        challenger=CHALLENGER,
        overlay_profile=OVERLAY,
    )

    assert result["status"] == "challenger_managed_promoted"
    assert result["base_primary_short_delta_r"] == 18.1
    assert result["pass_stress_primary_pf"] is True
    assert result["regime_specialist_tendency_persists"] is False


def test_build_promotion_decision_retains_champion_when_stress_pf_fails() -> None:
    summary_rows = [
        make_summary_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHAMPION, cum_r=9.1, pf=1.06, max_dd_r=21.2),
        make_summary_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHALLENGER, cum_r=27.2, pf=1.20, max_dd_r=13.4),
        make_summary_row(cost_scenario="base", window=SECONDARY_WINDOW, strategy_profile=CHAMPION, cum_r=13.5, pf=1.07, max_dd_r=9.4),
        make_summary_row(cost_scenario="base", window=SECONDARY_WINDOW, strategy_profile=CHALLENGER, cum_r=13.5, pf=1.07, max_dd_r=9.4),
        make_summary_row(cost_scenario="stress_x2", window=PRIMARY_WINDOW, strategy_profile=CHAMPION, cum_r=5.8, pf=1.02, max_dd_r=22.8),
        make_summary_row(cost_scenario="stress_x2", window=PRIMARY_WINDOW, strategy_profile=CHALLENGER, cum_r=21.1, pf=1.00, max_dd_r=14.9),
    ]
    side_rows = [
        make_side_row(cost_scenario="base", window=SECONDARY_WINDOW, strategy_profile=CHAMPION, side="LONG", cum_r=13.5),
        make_side_row(cost_scenario="base", window=SECONDARY_WINDOW, strategy_profile=CHALLENGER, side="LONG", cum_r=13.5),
        make_side_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHAMPION, side="LONG", cum_r=12.8),
        make_side_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHALLENGER, side="LONG", cum_r=12.8),
        make_side_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHAMPION, side="SHORT", cum_r=-3.7),
        make_side_row(cost_scenario="base", window=PRIMARY_WINDOW, strategy_profile=CHALLENGER, side="SHORT", cum_r=14.4),
    ]

    result = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=[],
        champion=CHAMPION,
        challenger=CHALLENGER,
        overlay_profile=OVERLAY,
    )

    assert result["status"] == "champion_managed_retained"
    assert result["pass_stress_primary_pf"] is False
    assert result["promoted_baseline_profile"] == CHAMPION


def test_build_concentration_summary_flags_regime_specialist_when_top3_share_is_concentrated() -> None:
    rows = [
        {"cost_scenario": "base", "horizon": "quarter", "label": "2022Q2", "strategy_profile": CHAMPION, "cum_r": 0.0},
        {"cost_scenario": "base", "horizon": "quarter", "label": "2022Q2", "strategy_profile": CHALLENGER, "cum_r": 10.0},
        {"cost_scenario": "base", "horizon": "quarter", "label": "2022Q3", "strategy_profile": CHAMPION, "cum_r": 0.0},
        {"cost_scenario": "base", "horizon": "quarter", "label": "2022Q3", "strategy_profile": CHALLENGER, "cum_r": 8.0},
        {"cost_scenario": "base", "horizon": "quarter", "label": "2026Q1", "strategy_profile": CHAMPION, "cum_r": 0.0},
        {"cost_scenario": "base", "horizon": "quarter", "label": "2026Q1", "strategy_profile": CHALLENGER, "cum_r": 7.0},
        {"cost_scenario": "base", "horizon": "quarter", "label": "2025Q1", "strategy_profile": CHAMPION, "cum_r": 0.0},
        {"cost_scenario": "base", "horizon": "quarter", "label": "2025Q1", "strategy_profile": CHALLENGER, "cum_r": 1.0},
    ]

    result = build_concentration_summary(rows=rows, champion=CHAMPION, challenger=CHALLENGER)

    assert result == [
        {
            "horizon": "quarter",
            "positive_windows": 4,
            "total_positive_delta_r": 26.0,
            "top_positive_label": "2022Q2",
            "top_positive_delta_r": 10.0,
            "top3_positive_share_pct": 96.15,
            "regime_specialist_tendency": True,
        }
    ]
