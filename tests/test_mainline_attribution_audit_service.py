from __future__ import annotations

import pandas as pd

from app.services.mainline_attribution_audit_service import MainlineAttributionAuditService


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_label": "two_year",
                "year": 2024,
                "side": "LONG",
                "trend_strength": 95,
                "bars_held": 10,
                "mfe_r": 2.0,
                "mae_r": 0.7,
                "mfe_capture_pct": 50.0,
                "tp1_hit": True,
                "tp2_hit": False,
                "exit_reason": "breakeven_after_tp1",
                "pnl_r": 1.0,
                "signal_time": pd.Timestamp("2024-03-01T00:00:00Z"),
            },
            {
                "window_label": "two_year",
                "year": 2024,
                "side": "LONG",
                "trend_strength": 92,
                "bars_held": 6,
                "mfe_r": 1.0,
                "mae_r": 1.1,
                "mfe_capture_pct": -50.0,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
                "pnl_r": -0.5,
                "signal_time": pd.Timestamp("2024-03-02T00:00:00Z"),
            },
            {
                "window_label": "two_year",
                "year": 2024,
                "side": "SHORT",
                "trend_strength": 88,
                "bars_held": 4,
                "mfe_r": 0.3,
                "mae_r": 1.0,
                "mfe_capture_pct": -333.33,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
                "pnl_r": -1.0,
                "signal_time": pd.Timestamp("2024-03-03T00:00:00Z"),
            },
            {
                "window_label": "two_year",
                "year": 2024,
                "side": "SHORT",
                "trend_strength": 90,
                "bars_held": 8,
                "mfe_r": 1.8,
                "mae_r": 0.4,
                "mfe_capture_pct": 66.67,
                "tp1_hit": True,
                "tp2_hit": True,
                "exit_reason": "take_profit_1.5R",
                "pnl_r": 1.2,
                "signal_time": pd.Timestamp("2024-03-04T00:00:00Z"),
            },
            {
                "window_label": "two_year",
                "year": 2024,
                "side": "LONG",
                "trend_strength": 93,
                "bars_held": 5,
                "mfe_r": 0.5,
                "mae_r": 1.2,
                "mfe_capture_pct": -160.0,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
                "pnl_r": -0.8,
                "signal_time": pd.Timestamp("2024-03-05T00:00:00Z"),
            },
            {
                "window_label": "two_year",
                "year": 2024,
                "side": "LONG",
                "trend_strength": 94,
                "bars_held": 3,
                "mfe_r": 0.2,
                "mae_r": 1.0,
                "mfe_capture_pct": -450.0,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
                "pnl_r": -0.9,
                "signal_time": pd.Timestamp("2024-03-06T00:00:00Z"),
            },
            {
                "window_label": "two_year",
                "year": 2024,
                "side": "LONG",
                "trend_strength": 96,
                "bars_held": 11,
                "mfe_r": 1.7,
                "mae_r": 0.5,
                "mfe_capture_pct": 70.59,
                "tp1_hit": True,
                "tp2_hit": False,
                "exit_reason": "breakeven_after_tp1",
                "pnl_r": 1.2,
                "signal_time": pd.Timestamp("2024-03-07T00:00:00Z"),
            },
        ]
    )


def test_summarize_groups_reports_core_trade_and_path_metrics() -> None:
    service = MainlineAttributionAuditService()
    frame = make_frame()

    rows = service.summarize_groups(frame, group_cols=["window_label", "side"])
    long_row = next(row for row in rows if row["side"] == "LONG")
    short_row = next(row for row in rows if row["side"] == "SHORT")

    assert long_row["trades"] == 5
    assert long_row["win_rate_pct"] == 40.0
    assert long_row["expectancy_r"] == 0.0
    assert long_row["cumulative_r"] == 0.0
    assert long_row["profit_factor"] == 1.0
    assert long_row["avg_mfe_r"] == 1.08
    assert long_row["avg_mae_r"] == 0.9

    assert short_row["trades"] == 2
    assert short_row["win_rate_pct"] == 50.0
    assert short_row["expectancy_r"] == 0.1
    assert short_row["cumulative_r"] == 0.2
    assert short_row["profit_factor"] == 1.2
    assert short_row["tp2_hit_rate_pct"] == 50.0


def test_loss_cluster_identification_and_summary_uses_consecutive_negative_runs() -> None:
    service = MainlineAttributionAuditService()
    frame = make_frame()

    cluster_rows = service.identify_loss_clusters(frame)
    summary_rows = service.summarize_loss_clusters(cluster_rows)

    assert len(cluster_rows) == 2
    first_cluster = cluster_rows[0]
    second_cluster = cluster_rows[1]
    assert first_cluster["cluster_length"] == 2
    assert first_cluster["cluster_cumulative_r"] == -1.5
    assert second_cluster["cluster_length"] == 2
    assert second_cluster["cluster_cumulative_r"] == -1.7
    assert first_cluster["short_count"] == 1
    assert second_cluster["long_count"] == 2
    assert second_cluster["short_count"] == 0

    summary = summary_rows[0]
    assert summary["window_label"] == "two_year"
    assert summary["clusters"] == 2
    assert summary["loss_trades_in_clusters"] == 4
    assert summary["max_cluster_length"] == 2
    assert summary["worst_cluster_cumulative_r"] == -1.7
    assert summary["worst_cluster_length"] == 2
