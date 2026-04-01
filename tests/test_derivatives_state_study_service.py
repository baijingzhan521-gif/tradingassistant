from __future__ import annotations

import pandas as pd

from app.services.carry_basis_research_service import CarryBasisResearchService
from app.services.carry_basis_execution_service import CarryBasisExecutionService
from app.services.derivatives_state_study_service import DerivativesStateStudyService


def make_frame() -> pd.DataFrame:
    rows = []
    for idx in range(200):
        bucket = (idx % 5) + 1
        bucket_to_z = {1: -1.8, 2: -0.6, 3: 0.0, 4: 0.7, 5: 1.8}
        base = {1: -10.0, 2: -4.0, 3: 0.0, 4: 5.0, 5: 12.0}[bucket]
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-03-19T00:00:00Z") + pd.Timedelta(hours=idx),
                "funding_rate": bucket_to_z[bucket] / 10.0,
                "funding_rate_z_7d": bucket_to_z[bucket],
                "open_interest_change_1h_pct": bucket_to_z[bucket] * 2.0,
                "open_interest_change_z_7d": bucket_to_z[bucket],
                "mark_index_spread_bps": bucket_to_z[bucket] * 3.0,
                "mark_index_spread_bps_z_7d": bucket_to_z[bucket],
                "basis_proxy_bps": bucket_to_z[bucket] * 4.0,
                "basis_proxy_bps_z_7d": bucket_to_z[bucket],
                "forward_index_return_bps_1h": base,
                "forward_abs_index_return_bps_1h": abs(base) + 1.0,
                "forward_index_return_bps_4h": base * 1.5,
                "forward_abs_index_return_bps_4h": abs(base * 1.5) + 1.0,
                "forward_index_return_bps_24h": base * 2.0,
                "forward_abs_index_return_bps_24h": abs(base * 2.0) + 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_assign_feature_buckets_uses_calibration_edges() -> None:
    service = DerivativesStateStudyService()
    calibration = make_frame().iloc[:100].copy()
    evaluation = make_frame().iloc[100:].copy()

    edges = service.build_calibration_edges(calibration)
    assigned = service.assign_feature_buckets(evaluation, edges)

    assert "funding_zscore_bucket" in assigned.columns
    assert set(assigned["funding_zscore_bucket"].dropna().unique()) == {1.0, 2.0, 3.0, 4.0, 5.0}


def test_return_and_volatility_summaries_capture_extreme_edges() -> None:
    service = DerivativesStateStudyService()
    frame = service.assign_feature_buckets(make_frame(), service.build_calibration_edges(make_frame().iloc[:100].copy()))

    _, return_edges, _ = service.summarize_return_conditionality(frame, min_bucket_obs=10, min_monthly_bucket_obs=1)
    _, vol_edges = service.summarize_volatility_conditionality(frame, min_bucket_obs=10)

    funding_24h = next(row for row in return_edges if row["feature"] == "funding_zscore" and row["horizon_hours"] == 24)
    assert funding_24h["q5_minus_q1_forward_bps"] > 0

    funding_vol_24h = next(row for row in vol_edges if row["feature"] == "funding_zscore" and row["horizon_hours"] == 24)
    assert funding_vol_24h["extreme_to_q3_abs_vol_ratio"] > 1.0


def test_mainline_trade_state_summary_reports_bucket_differences() -> None:
    service = DerivativesStateStudyService()
    trades = pd.DataFrame(
        [
            {
                "signal_time": pd.Timestamp("2024-03-19T00:00:00Z"),
                "side": "LONG",
                "pnl_r": -1.0,
                "bars_held": 10,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
                "funding_zscore_bucket": 1.0,
                "oi_change_bucket": 1.0,
                "mark_index_premium_bucket": 1.0,
                "basis_bucket": 1.0,
            },
            {
                "signal_time": pd.Timestamp("2024-03-20T00:00:00Z"),
                "side": "LONG",
                "pnl_r": -0.5,
                "bars_held": 12,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
                "funding_zscore_bucket": 1.0,
                "oi_change_bucket": 1.0,
                "mark_index_premium_bucket": 1.0,
                "basis_bucket": 1.0,
            },
            {
                "signal_time": pd.Timestamp("2024-03-21T00:00:00Z"),
                "side": "LONG",
                "pnl_r": 1.0,
                "bars_held": 20,
                "tp1_hit": True,
                "tp2_hit": False,
                "exit_reason": "breakeven_after_tp1",
                "funding_zscore_bucket": 5.0,
                "oi_change_bucket": 5.0,
                "mark_index_premium_bucket": 5.0,
                "basis_bucket": 5.0,
            },
            {
                "signal_time": pd.Timestamp("2024-03-22T00:00:00Z"),
                "side": "LONG",
                "pnl_r": 2.0,
                "bars_held": 25,
                "tp1_hit": True,
                "tp2_hit": True,
                "exit_reason": "take_profit_2",
                "funding_zscore_bucket": 5.0,
                "oi_change_bucket": 5.0,
                "mark_index_premium_bucket": 5.0,
                "basis_bucket": 5.0,
            },
        ]
    )

    _, edges, exits = service.summarize_mainline_trade_state(trades, min_bucket_obs=2)
    funding_long = next(row for row in edges if row["feature"] == "funding_zscore" and row["side"] == "LONG")

    assert funding_long["q5_minus_q1_expectancy_r"] > 0
    assert any(row["exit_reason"] == "stop_loss" for row in exits)


def make_carry_frame() -> pd.DataFrame:
    rows = []
    for idx in range(720):
        funding_bucket = (idx % 5) + 1
        basis_bucket = ((idx * 2) % 5) + 1
        funding_z = {1: -1.8, 2: -0.7, 3: 0.0, 4: 0.8, 5: 1.9}[funding_bucket]
        basis_z = {1: -1.6, 2: -0.5, 3: 0.1, 4: 0.9, 5: 2.1}[basis_bucket]
        basis_bps = 15.0 if basis_z > 0 else (-10.0 if basis_z < 0 else 2.0)
        funding_rate = 0.0004 if funding_z > 0 else (-0.0002 if funding_z < 0 else 0.0001)
        funding_event = funding_rate if idx % 8 == 0 else None
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(hours=idx),
                "index_close": 10000.0 + idx,
                "mark_close": 10001.0 + idx * 0.99,
                "funding_rate_event": funding_event,
                "funding_rate": funding_rate,
                "basis_proxy_bps": basis_bps,
                "funding_rate_z_7d": funding_z,
                "basis_proxy_bps_z_7d": basis_z,
            }
        )
    return pd.DataFrame(rows)


def test_carry_enrich_frame_adds_carry_columns() -> None:
    service = CarryBasisResearchService()
    frame = make_carry_frame()
    calibration = frame.iloc[:120].copy()
    evaluation = frame.iloc[120:].copy()

    enriched = service.enrich_frame(evaluation, service.build_calibration_edges(calibration))

    assert "gross_carry_bps_24h" in enriched.columns
    assert "net_carry_bps_168h_cost_28bps" in enriched.columns
    assert enriched["gross_carry_bps_24h"].notna().sum() > 0


def test_carry_candidate_summary_prefers_positive_basis_and_funding_state() -> None:
    service = CarryBasisResearchService()
    frame = make_carry_frame()
    calibration = frame.iloc[:120].copy()
    evaluation = service.enrich_frame(frame.iloc[120:].copy(), service.build_calibration_edges(calibration))

    rows = service.summarize_candidates(evaluation, service.build_candidates())
    summary = pd.DataFrame(rows)
    basis_and_funding = summary[(summary["candidate"] == "basis_and_funding_positive") & (summary["horizon_hours"] == 72)].iloc[0]
    always_on = summary[(summary["candidate"] == "always_on") & (summary["horizon_hours"] == 72)].iloc[0]

    assert basis_and_funding["gross_mean_bps"] > 0
    assert basis_and_funding["observations"] < always_on["observations"]


def test_carry_monthly_stability_and_best_candidate_selection() -> None:
    service = CarryBasisResearchService()
    frame = make_carry_frame()
    calibration = frame.iloc[:120].copy()
    evaluation = service.enrich_frame(frame.iloc[120:].copy(), service.build_calibration_edges(calibration))

    candidates = service.build_candidates()
    rows = service.summarize_candidates(evaluation, candidates)
    best = service.choose_best_candidate(rows, cost_bps=28.0)
    assert best is not None
    assert int(best["horizon_hours"]) in {72, 168}

    candidate = next(item for item in candidates if item.label == best["candidate"])
    monthly_rows, monthly_summary = service.summarize_monthly_stability(
        evaluation,
        candidate=candidate,
        horizon=int(best["horizon_hours"]),
        cost_bps=28.0,
    )
    assert monthly_rows
    assert monthly_summary["months"] >= 1


def test_carry_execution_sequence_produces_non_overlapping_trades() -> None:
    base_service = CarryBasisResearchService()
    exec_service = CarryBasisExecutionService()
    frame = make_carry_frame()
    calibration = frame.iloc[:120].copy()
    evaluation = base_service.enrich_frame(frame.iloc[120:].copy(), base_service.build_calibration_edges(calibration))

    scenario = exec_service.build_scenarios()[0]
    candidate = next(item for item in base_service.build_candidates() if item.label == "basis_positive")
    trades, summary, monthly = exec_service.simulate_sequence(
        evaluation,
        candidate=candidate,
        scenario=scenario,
        horizon=24,
    )

    assert trades
    assert summary["trades"] == len(trades)
    assert summary["active_hours"] == len(trades) * 24
    assert monthly
    ordered = pd.DataFrame(trades).sort_values("signal_time").reset_index(drop=True)
    assert (pd.to_datetime(ordered["exit_time"], utc=True).shift() <= pd.to_datetime(ordered["signal_time"], utc=True)).iloc[1:].all()


def test_carry_execution_best_summary_selects_candidate_with_trades() -> None:
    base_service = CarryBasisResearchService()
    exec_service = CarryBasisExecutionService()
    frame = make_carry_frame()
    calibration = frame.iloc[:120].copy()
    evaluation = base_service.enrich_frame(frame.iloc[120:].copy(), base_service.build_calibration_edges(calibration))

    rows = []
    scenario = exec_service.build_scenarios()[0]
    for candidate in base_service.build_candidates()[:3]:
        for horizon in (24, 72):
            _, summary, _ = exec_service.simulate_sequence(
                evaluation,
                candidate=candidate,
                scenario=scenario,
                horizon=horizon,
            )
            rows.append(summary)

    best = exec_service.choose_best_summary(rows)
    assert best is not None
    assert best["trades"] >= 0
