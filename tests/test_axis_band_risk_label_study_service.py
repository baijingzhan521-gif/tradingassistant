from __future__ import annotations

import pandas as pd

from app.services.axis_band_risk_label_study_service import AxisBandRiskLabelStudyService


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_label": "full_2020",
                "side": "LONG",
                "signal_time": pd.Timestamp("2024-03-01T00:00:00Z"),
                "pnl_r": -0.8,
                "bars_held": 6,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
                "risk_group": "active",
                "risk_label": "pullback_risk",
                "risk_extreme": True,
                "forward_close_r_4h": -0.4,
                "forward_close_r_24h": -0.9,
                "forward_mfe_r_4h": 0.2,
                "forward_mfe_r_24h": 0.5,
                "forward_mae_r_4h": 0.8,
                "forward_mae_r_24h": 1.1,
            },
            {
                "window_label": "full_2020",
                "side": "LONG",
                "signal_time": pd.Timestamp("2024-03-10T00:00:00Z"),
                "pnl_r": 0.6,
                "bars_held": 10,
                "tp1_hit": True,
                "tp2_hit": False,
                "exit_reason": "breakeven_after_tp1",
                "risk_group": "inactive",
                "risk_label": "none",
                "risk_extreme": False,
                "forward_close_r_4h": 0.2,
                "forward_close_r_24h": 0.8,
                "forward_mfe_r_4h": 0.7,
                "forward_mfe_r_24h": 1.3,
                "forward_mae_r_4h": 0.3,
                "forward_mae_r_24h": 0.5,
            },
            {
                "window_label": "full_2020",
                "side": "SHORT",
                "signal_time": pd.Timestamp("2024-04-01T00:00:00Z"),
                "pnl_r": -0.7,
                "bars_held": 7,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
                "risk_group": "active",
                "risk_label": "rebound_risk",
                "risk_extreme": False,
                "forward_close_r_4h": -0.3,
                "forward_close_r_24h": -0.6,
                "forward_mfe_r_4h": 0.2,
                "forward_mfe_r_24h": 0.4,
                "forward_mae_r_4h": 0.7,
                "forward_mae_r_24h": 0.9,
            },
            {
                "window_label": "full_2020",
                "side": "SHORT",
                "signal_time": pd.Timestamp("2024-04-12T00:00:00Z"),
                "pnl_r": 0.4,
                "bars_held": 9,
                "tp1_hit": True,
                "tp2_hit": False,
                "exit_reason": "breakeven_after_tp1",
                "risk_group": "inactive",
                "risk_label": "none",
                "risk_extreme": False,
                "forward_close_r_4h": 0.1,
                "forward_close_r_24h": 0.5,
                "forward_mfe_r_4h": 0.6,
                "forward_mfe_r_24h": 1.0,
                "forward_mae_r_4h": 0.3,
                "forward_mae_r_24h": 0.4,
            },
        ]
    )


def test_trade_and_path_summaries_keep_side_specific_labels() -> None:
    service = AxisBandRiskLabelStudyService()
    frame = make_frame()

    trade_rows = service.summarize_trade_distribution(frame)
    path_rows = service.summarize_path_distribution(frame)

    long_active = next(row for row in trade_rows if row["side"] == "LONG" and row["risk_group"] == "active")
    short_active = next(row for row in trade_rows if row["side"] == "SHORT" and row["risk_group"] == "active")
    assert long_active["risk_label"] == "pullback_risk"
    assert short_active["risk_label"] == "rebound_risk"

    long_path = next(row for row in path_rows if row["side"] == "LONG" and row["risk_group"] == "active")
    assert long_path["mean_forward_close_r_24h"] < 0
    assert long_path["mean_mae_r_24h"] > 0


def test_edge_and_monthly_stability_report_active_minus_inactive() -> None:
    service = AxisBandRiskLabelStudyService()
    frame = pd.concat([make_frame(), make_frame()], ignore_index=True)
    frame.loc[4:, "signal_time"] = frame.loc[4:, "signal_time"] + pd.Timedelta(days=31)

    edge_rows = service.summarize_edge(frame)
    _, stability_rows = service.summarize_monthly_stability(frame, min_obs_per_group=1)

    long_edge = next(row for row in edge_rows if row["side"] == "LONG")
    assert long_edge["delta_expectancy_r"] < 0
    assert long_edge["delta_forward_close_r_24h"] < 0
    assert long_edge["delta_mae_r_24h"] > 0

    long_stability = next(row for row in stability_rows if row["side"] == "LONG" and row["metric"] == "pnl_r")
    assert long_stability["months_with_both_groups"] >= 2
    assert long_stability["same_sign_month_rate_pct"] == 100.0
