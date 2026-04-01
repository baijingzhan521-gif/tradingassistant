from __future__ import annotations

import pandas as pd

from app.services.derivatives_predictive_power_service import DerivativesPredictivePowerService


def make_frame() -> pd.DataFrame:
    rows = []
    for idx in range(90):
        z = -1.5 if idx < 30 else (1.5 if idx >= 60 else 0.0)
        base_forward = -10.0 if idx < 30 else (10.0 if idx >= 60 else 0.0)
        rows.append(
            {
                "funding_rate_z_7d": z,
                "open_interest_change_1h_pct": z,
                "open_interest_change_z_7d": z,
                "mark_index_spread_bps": z * 5.0,
                "mark_index_spread_bps_z_7d": z,
                "basis_proxy_bps": z * 4.0,
                "basis_proxy_bps_z_7d": z,
                "forward_index_return_bps_1h": base_forward,
                "forward_abs_index_return_bps_1h": abs(base_forward) + 1.0,
                "forward_index_return_bps_4h": base_forward * 2.0,
                "forward_abs_index_return_bps_4h": abs(base_forward * 2.0) + 1.0,
                "forward_index_return_bps_24h": base_forward * 3.0,
                "forward_abs_index_return_bps_24h": abs(base_forward * 3.0) + 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_summarize_availability_marks_liquidation_unavailable() -> None:
    service = DerivativesPredictivePowerService()
    rows = service.summarize_availability(make_frame())
    lookup = {row["feature"]: row for row in rows}

    assert lookup["funding_zscore"]["available"] is True
    assert lookup["liquidation_burst"]["available"] is False


def test_summarize_state_tables_and_edges_capture_directional_split() -> None:
    service = DerivativesPredictivePowerService()
    state_rows = service.summarize_state_tables(make_frame())
    edges = service.summarize_feature_edges(state_rows)

    funding_1h = [row for row in state_rows if row["feature"] == "funding_zscore" and row["horizon_hours"] == 1]
    lookup = {row["state"]: row for row in funding_1h}
    assert lookup["low"]["mean_forward_bps"] < lookup["mid"]["mean_forward_bps"] < lookup["high"]["mean_forward_bps"]

    edge_lookup = {
        (row["feature"], row["horizon_hours"]): row
        for row in edges
    }
    assert edge_lookup[("funding_zscore", 1)]["high_minus_low_forward_bps"] > 0


def test_summarize_correlations_returns_non_empty_rows() -> None:
    service = DerivativesPredictivePowerService()
    rows = service.summarize_correlations(make_frame())

    assert any(row["feature"] == "basis" and row["horizon_hours"] == 24 for row in rows)
