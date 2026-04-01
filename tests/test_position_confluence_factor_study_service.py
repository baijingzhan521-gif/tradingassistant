from __future__ import annotations

import pandas as pd

from app.services.position_confluence_factor_study_service import PositionConfluenceFactorStudyService


def make_factor_frame() -> pd.DataFrame:
    rows = []
    for idx in range(250):
        bucket = (idx % 5) + 1
        value = {1: 0.10, 2: 0.35, 3: 0.60, 4: 0.90, 5: 1.25}[bucket]
        base = {1: 12.0, 2: 8.0, 3: 3.0, 4: -2.0, 5: -7.0}[bucket]
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-03-19T00:00:00Z") + pd.Timedelta(hours=idx),
                "ema55_anchor_gap_atr": value,
                "pivot_anchor_gap_atr": value + 0.05,
                "band_anchor_gap_atr": value + 0.10,
                "confluence_spread_atr": value + 0.15,
                "aligned_forward_return_bps_1h": base,
                "aligned_forward_return_bps_4h": base * 1.5,
                "aligned_forward_return_bps_24h": base * 2.0,
                "forward_abs_return_bps_1h": abs(base) + 1.0,
                "forward_abs_return_bps_4h": abs(base * 1.5) + 1.0,
                "forward_abs_return_bps_24h": abs(base * 2.0) + 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_assign_factor_buckets_uses_calibration_edges() -> None:
    service = PositionConfluenceFactorStudyService()
    calibration = make_factor_frame().iloc[:125].copy()
    evaluation = make_factor_frame().iloc[125:].copy()

    edges = service.build_calibration_edges(calibration)
    assigned = service.assign_factor_buckets(evaluation, edges)

    assert "ema55_anchor_gap_bucket" in assigned.columns
    assert set(assigned["ema55_anchor_gap_bucket"].dropna().unique()) == {1.0, 2.0, 3.0, 4.0, 5.0}


def test_return_and_volatility_summaries_capture_tighter_is_better_pattern() -> None:
    service = PositionConfluenceFactorStudyService()
    frame = service.assign_factor_buckets(make_factor_frame(), service.build_calibration_edges(make_factor_frame().iloc[:125].copy()))

    _, return_edges, _ = service.summarize_return_conditionality(frame, min_bucket_obs=10, min_monthly_bucket_obs=1)
    _, vol_edges = service.summarize_volatility_conditionality(frame, min_bucket_obs=10)

    ema55_24h = next(row for row in return_edges if row["factor"] == "ema55_anchor_gap" and row["horizon_hours"] == 24)
    assert ema55_24h["q1_minus_q5_aligned_bps"] > 0
    assert ema55_24h["monotonic_pair_rate_pct"] > 0

    ema55_vol_24h = next(row for row in vol_edges if row["factor"] == "ema55_anchor_gap" and row["horizon_hours"] == 24)
    assert ema55_vol_24h["extreme_to_q3_abs_vol_ratio"] > 1.0


def test_mainline_trade_factor_summary_reports_bucket_differences() -> None:
    service = PositionConfluenceFactorStudyService()
    trades = pd.DataFrame(
        [
            {
                "signal_time": pd.Timestamp("2024-03-19T00:00:00Z"),
                "side": "LONG",
                "pnl_r": -1.0,
                "bars_held": 10,
                "exit_reason": "stop_loss",
                "ema55_anchor_gap_bucket": 5.0,
                "pivot_anchor_gap_bucket": 5.0,
                "band_anchor_gap_bucket": 5.0,
                "confluence_spread_bucket": 5.0,
            },
            {
                "signal_time": pd.Timestamp("2024-03-20T00:00:00Z"),
                "side": "LONG",
                "pnl_r": -0.5,
                "bars_held": 11,
                "exit_reason": "stop_loss",
                "ema55_anchor_gap_bucket": 5.0,
                "pivot_anchor_gap_bucket": 5.0,
                "band_anchor_gap_bucket": 5.0,
                "confluence_spread_bucket": 5.0,
            },
            {
                "signal_time": pd.Timestamp("2024-03-21T00:00:00Z"),
                "side": "LONG",
                "pnl_r": 1.0,
                "bars_held": 20,
                "exit_reason": "breakeven_after_tp1",
                "ema55_anchor_gap_bucket": 1.0,
                "pivot_anchor_gap_bucket": 1.0,
                "band_anchor_gap_bucket": 1.0,
                "confluence_spread_bucket": 1.0,
            },
            {
                "signal_time": pd.Timestamp("2024-03-22T00:00:00Z"),
                "side": "LONG",
                "pnl_r": 2.0,
                "bars_held": 25,
                "exit_reason": "tp2",
                "ema55_anchor_gap_bucket": 1.0,
                "pivot_anchor_gap_bucket": 1.0,
                "band_anchor_gap_bucket": 1.0,
                "confluence_spread_bucket": 1.0,
            },
        ]
    )

    _, edges, exits = service.summarize_mainline_trade_factors(trades, min_bucket_obs=2)
    ema55_long = next(row for row in edges if row["factor"] == "ema55_anchor_gap" and row["side"] == "LONG")

    assert ema55_long["q1_minus_q5_expectancy_r"] > 0
    assert any(row["exit_reason"] == "stop_loss" for row in exits)
