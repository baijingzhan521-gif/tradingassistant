from __future__ import annotations

from typing import Any

import pandas as pd


PATH_HORIZONS = (4, 24)
MONTHLY_METRICS = ("pnl_r", "forward_close_r_4h", "forward_close_r_24h")


class AxisBandRiskLabelStudyService:
    def summarize_trade_distribution(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if frame.empty:
            return rows

        grouped = frame.groupby(["window_label", "side", "risk_group"], sort=True)
        for (window_label, side, risk_group), group in grouped:
            rows.append(
                {
                    "window_label": window_label,
                    "side": side,
                    "risk_group": risk_group,
                    "risk_label": self._risk_label_for_group(side=side, risk_group=risk_group),
                    "trades": int(len(group)),
                    "win_rate_pct": round(float((group["pnl_r"] > 0).mean() * 100.0), 2),
                    "expectancy_r": round(float(group["pnl_r"].mean()), 4),
                    "cumulative_r": round(float(group["pnl_r"].sum()), 4),
                    "avg_bars_held": round(float(group["bars_held"].mean()), 2),
                    "tp1_hit_rate_pct": round(float(group["tp1_hit"].mean() * 100.0), 2),
                    "tp2_hit_rate_pct": round(float(group["tp2_hit"].mean() * 100.0), 2),
                    "stop_loss_rate_pct": round(float((group["exit_reason"] == "stop_loss").mean() * 100.0), 2),
                    "breakeven_rate_pct": round(
                        float((group["exit_reason"] == "breakeven_after_tp1").mean() * 100.0),
                        2,
                    ),
                    "high_severity_rate_pct": round(float(group["risk_extreme"].mean() * 100.0), 2),
                }
            )
        return rows

    def summarize_path_distribution(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if frame.empty:
            return rows

        grouped = frame.groupby(["window_label", "side", "risk_group"], sort=True)
        for (window_label, side, risk_group), group in grouped:
            row: dict[str, Any] = {
                "window_label": window_label,
                "side": side,
                "risk_group": risk_group,
                "risk_label": self._risk_label_for_group(side=side, risk_group=risk_group),
                "trades": int(len(group)),
            }
            for horizon in PATH_HORIZONS:
                close_col = f"forward_close_r_{horizon}h"
                mfe_col = f"forward_mfe_r_{horizon}h"
                mae_col = f"forward_mae_r_{horizon}h"
                subset = group[[close_col, mfe_col, mae_col]].dropna()
                row[f"obs_{horizon}h"] = int(len(subset))
                row[f"mean_forward_close_r_{horizon}h"] = round(float(subset[close_col].mean()), 4) if not subset.empty else None
                row[f"median_forward_close_r_{horizon}h"] = round(float(subset[close_col].median()), 4) if not subset.empty else None
                row[f"positive_forward_rate_pct_{horizon}h"] = (
                    round(float((subset[close_col] > 0).mean() * 100.0), 2) if not subset.empty else None
                )
                row[f"mean_mfe_r_{horizon}h"] = round(float(subset[mfe_col].mean()), 4) if not subset.empty else None
                row[f"mean_mae_r_{horizon}h"] = round(float(subset[mae_col].mean()), 4) if not subset.empty else None
            rows.append(row)
        return rows

    def summarize_edge(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if frame.empty:
            return rows

        for (window_label, side), subset in frame.groupby(["window_label", "side"], sort=True):
            active = subset[subset["risk_group"] == "active"]
            inactive = subset[subset["risk_group"] == "inactive"]
            if active.empty or inactive.empty:
                continue

            row: dict[str, Any] = {
                "window_label": window_label,
                "side": side,
                "risk_label": self._risk_label_for_group(side=side, risk_group="active"),
                "active_trades": int(len(active)),
                "inactive_trades": int(len(inactive)),
                "active_share_pct": round(float((len(active) / len(subset)) * 100.0), 2),
                "delta_expectancy_r": round(float(active["pnl_r"].mean() - inactive["pnl_r"].mean()), 4),
                "delta_win_rate_pct": round(float(((active["pnl_r"] > 0).mean() - (inactive["pnl_r"] > 0).mean()) * 100.0), 2),
                "delta_avg_bars_held": round(float(active["bars_held"].mean() - inactive["bars_held"].mean()), 2),
                "delta_stop_loss_rate_pct": round(
                    float(((active["exit_reason"] == "stop_loss").mean() - (inactive["exit_reason"] == "stop_loss").mean()) * 100.0),
                    2,
                ),
                "delta_breakeven_rate_pct": round(
                    float(
                        (
                            (active["exit_reason"] == "breakeven_after_tp1").mean()
                            - (inactive["exit_reason"] == "breakeven_after_tp1").mean()
                        )
                        * 100.0
                    ),
                    2,
                ),
                "delta_tp2_hit_rate_pct": round(float((active["tp2_hit"].mean() - inactive["tp2_hit"].mean()) * 100.0), 2),
                "active_high_severity_rate_pct": round(float(active["risk_extreme"].mean() * 100.0), 2),
            }
            for horizon in PATH_HORIZONS:
                close_col = f"forward_close_r_{horizon}h"
                mfe_col = f"forward_mfe_r_{horizon}h"
                mae_col = f"forward_mae_r_{horizon}h"
                active_path = active[[close_col, mfe_col, mae_col]].dropna()
                inactive_path = inactive[[close_col, mfe_col, mae_col]].dropna()
                if active_path.empty or inactive_path.empty:
                    row[f"delta_forward_close_r_{horizon}h"] = None
                    row[f"delta_mfe_r_{horizon}h"] = None
                    row[f"delta_mae_r_{horizon}h"] = None
                    continue
                row[f"delta_forward_close_r_{horizon}h"] = round(
                    float(active_path[close_col].mean() - inactive_path[close_col].mean()),
                    4,
                )
                row[f"delta_mfe_r_{horizon}h"] = round(float(active_path[mfe_col].mean() - inactive_path[mfe_col].mean()), 4)
                row[f"delta_mae_r_{horizon}h"] = round(float(active_path[mae_col].mean() - inactive_path[mae_col].mean()), 4)
            rows.append(row)
        return rows

    def summarize_monthly_stability(
        self,
        frame: pd.DataFrame,
        *,
        min_obs_per_group: int = 1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        monthly_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        if frame.empty:
            return monthly_rows, summary_rows

        enriched = frame.copy()
        enriched["signal_month"] = pd.to_datetime(enriched["signal_time"], utc=True).dt.strftime("%Y-%m")

        for (window_label, side), subset in enriched.groupby(["window_label", "side"], sort=True):
            active = subset[subset["risk_group"] == "active"]
            inactive = subset[subset["risk_group"] == "inactive"]
            if active.empty or inactive.empty:
                continue
            for metric in MONTHLY_METRICS:
                overall_diff = float(active[metric].mean() - inactive[metric].mean())
                month_group = (
                    subset.groupby(["signal_month", "risk_group"])[metric]
                    .agg(["mean", "size"])
                    .reset_index()
                )
                stable_months = 0
                same_sign_months = 0
                for month, month_frame in month_group.groupby("signal_month", sort=True):
                    active_row = month_frame[month_frame["risk_group"] == "active"]
                    inactive_row = month_frame[month_frame["risk_group"] == "inactive"]
                    if active_row.empty or inactive_row.empty:
                        continue
                    active_count = int(active_row["size"].iloc[0])
                    inactive_count = int(inactive_row["size"].iloc[0])
                    if active_count < min_obs_per_group or inactive_count < min_obs_per_group:
                        continue
                    month_diff = float(active_row["mean"].iloc[0] - inactive_row["mean"].iloc[0])
                    stable_months += 1
                    if overall_diff == 0.0 or month_diff == 0.0:
                        same_sign_months += 1
                    elif overall_diff > 0 and month_diff > 0:
                        same_sign_months += 1
                    elif overall_diff < 0 and month_diff < 0:
                        same_sign_months += 1
                    monthly_rows.append(
                        {
                            "window_label": window_label,
                            "side": side,
                            "metric": metric,
                            "month": month,
                            "active_obs": active_count,
                            "inactive_obs": inactive_count,
                            "active_minus_inactive": round(month_diff, 4),
                        }
                    )
                summary_rows.append(
                    {
                        "window_label": window_label,
                        "side": side,
                        "metric": metric,
                        "overall_active_minus_inactive": round(overall_diff, 4),
                        "months_with_both_groups": stable_months,
                        "same_sign_months": same_sign_months,
                        "same_sign_month_rate_pct": round((same_sign_months / stable_months) * 100.0, 2)
                        if stable_months
                        else 0.0,
                    }
                )
        return monthly_rows, summary_rows

    @staticmethod
    def _risk_label_for_group(*, side: str, risk_group: str) -> str:
        if risk_group == "inactive":
            return "none"
        return "pullback_risk" if side == "LONG" else "rebound_risk"
