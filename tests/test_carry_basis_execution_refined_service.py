from __future__ import annotations

import math

import pandas as pd

from app.services.carry_basis_execution_refined_service import CarryBasisExecutionRefinedService
from app.services.carry_basis_execution_service import CarryBasisExecutionService
from app.services.carry_basis_research_service import CarryBasisResearchService
from tests.test_derivatives_state_study_service import make_carry_frame


def test_refined_legacy_proxy_baseline_matches_original_sequence_summary() -> None:
    base_service = CarryBasisResearchService()
    legacy_service = CarryBasisExecutionService()
    refined_service = CarryBasisExecutionRefinedService()
    frame = make_carry_frame()
    calibration = frame.iloc[:120].copy()
    evaluation = base_service.enrich_frame(frame.iloc[120:].copy(), base_service.build_calibration_edges(calibration))
    candidate = next(item for item in base_service.build_candidates() if item.label == "basis_positive")

    legacy_scenario = next(item for item in legacy_service.build_scenarios() if item.label == "hybrid_maker_taker_15im")
    refined_scenario = next(item for item in refined_service.build_focus_scenarios() if item.label == "legacy_proxy_baseline")

    legacy_trades, legacy_summary, _ = legacy_service.simulate_sequence(
        evaluation,
        candidate=candidate,
        scenario=legacy_scenario,
        horizon=168,
    )
    refined_trades, refined_summary, _ = refined_service.simulate_sequence(
        evaluation,
        candidate=candidate,
        scenario=refined_scenario,
        horizon=168,
    )

    assert len(legacy_trades) == len(refined_trades)
    assert refined_summary["trades"] == legacy_summary["trades"]
    assert math.isclose(refined_summary["annualized_roc_pct"], legacy_summary["annualized_roc_pct"], abs_tol=1e-4)
    assert math.isclose(refined_summary["net_mean_bps"], legacy_summary["net_mean_bps"], abs_tol=1e-4)


def test_opportunity_cost_bps_scales_with_horizon_and_capital_mode() -> None:
    refined_service = CarryBasisExecutionRefinedService()
    pooled = next(item for item in refined_service.build_focus_scenarios() if item.label == "realistic_base_pooled_4opp")
    segregated = next(item for item in refined_service.build_focus_scenarios() if item.label == "realistic_base_segregated_4opp")

    pooled_cost = pooled.opportunity_cost_bps(168)
    segregated_cost = segregated.opportunity_cost_bps(168)

    expected_pooled = 4.0 * 100.0 * (168.0 / (24.0 * 365.0))
    expected_segregated = expected_pooled * 1.15
    assert math.isclose(pooled_cost, expected_pooled, rel_tol=1e-9)
    assert math.isclose(segregated_cost, expected_segregated, rel_tol=1e-9)
    assert segregated_cost > pooled_cost


def test_pooled_capital_mode_improves_roc_relative_to_segregated_when_other_costs_match() -> None:
    base_service = CarryBasisResearchService()
    refined_service = CarryBasisExecutionRefinedService()
    frame = make_carry_frame()
    calibration = frame.iloc[:120].copy()
    evaluation = base_service.enrich_frame(frame.iloc[120:].copy(), base_service.build_calibration_edges(calibration))
    candidate = next(item for item in base_service.build_candidates() if item.label == "basis_and_funding_positive")

    pooled = next(item for item in refined_service.build_focus_scenarios() if item.label == "realistic_base_pooled_4opp")
    segregated = next(item for item in refined_service.build_focus_scenarios() if item.label == "realistic_base_segregated_4opp")

    _, pooled_summary, _ = refined_service.simulate_sequence(
        evaluation,
        candidate=candidate,
        scenario=pooled,
        horizon=168,
    )
    _, segregated_summary, _ = refined_service.simulate_sequence(
        evaluation,
        candidate=candidate,
        scenario=segregated,
        horizon=168,
    )

    assert pooled_summary["trades"] == segregated_summary["trades"]
    assert pooled_summary["all_in_cost_bps"] < segregated_summary["all_in_cost_bps"]
    assert pooled_summary["annualized_roc_pct"] > segregated_summary["annualized_roc_pct"]


def test_refined_sequence_remains_non_overlapping() -> None:
    base_service = CarryBasisResearchService()
    refined_service = CarryBasisExecutionRefinedService()
    frame = make_carry_frame()
    calibration = frame.iloc[:120].copy()
    evaluation = base_service.enrich_frame(frame.iloc[120:].copy(), base_service.build_calibration_edges(calibration))
    candidate = next(item for item in base_service.build_candidates() if item.label == "basis_positive")
    scenario = next(item for item in refined_service.build_focus_scenarios() if item.label == "realistic_base_segregated_8opp")

    trades, summary, monthly = refined_service.simulate_sequence(
        evaluation,
        candidate=candidate,
        scenario=scenario,
        horizon=168,
    )

    assert trades
    assert summary["trades"] == len(trades)
    assert monthly
    ordered = pd.DataFrame(trades).sort_values("signal_time").reset_index(drop=True)
    assert (pd.to_datetime(ordered["exit_time"], utc=True).shift() <= pd.to_datetime(ordered["signal_time"], utc=True)).iloc[1:].all()
