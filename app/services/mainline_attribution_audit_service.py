from __future__ import annotations

from typing import Any

import pandas as pd


class MainlineAttributionAuditService:
    def summarize_groups(self, frame: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if frame.empty:
            return rows

        grouped = frame.groupby(group_cols, sort=True, dropna=False, observed=True)
        for keys, group in grouped:
            keys = (keys,) if not isinstance(keys, tuple) else keys
            row = {column: value for column, value in zip(group_cols, keys)}

            wins = group[group["pnl_r"] > 0]["pnl_r"].sum()
            losses = -group[group["pnl_r"] < 0]["pnl_r"].sum()
            profit_factor = float(wins / losses) if losses > 0 else None

            row.update(
                {
                    "trades": int(len(group)),
                    "win_rate_pct": round(float((group["pnl_r"] > 0).mean() * 100.0), 2),
                    "expectancy_r": round(float(group["pnl_r"].mean()), 4),
                    "cumulative_r": round(float(group["pnl_r"].sum()), 4),
                    "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
                    "avg_trend_strength": round(float(group["trend_strength"].mean()), 2),
                    "avg_bars_held": round(float(group["bars_held"].mean()), 2),
                    "median_bars_held": round(float(group["bars_held"].median()), 2),
                    "avg_mfe_r": round(float(group["mfe_r"].mean()), 4),
                    "avg_mae_r": round(float(group["mae_r"].mean()), 4),
                    "median_mfe_r": round(float(group["mfe_r"].median()), 4),
                    "median_mae_r": round(float(group["mae_r"].median()), 4),
                    "avg_mfe_capture_pct": round(float(group["mfe_capture_pct"].mean()), 2),
                    "tp1_hit_rate_pct": round(float(group["tp1_hit"].mean() * 100.0), 2),
                    "tp2_hit_rate_pct": round(float(group["tp2_hit"].mean() * 100.0), 2),
                    "stop_loss_rate_pct": round(float((group["exit_reason"] == "stop_loss").mean() * 100.0), 2),
                    "breakeven_rate_pct": round(
                        float((group["exit_reason"] == "breakeven_after_tp1").mean() * 100.0),
                        2,
                    ),
                }
            )
            rows.append(row)
        return rows

    def identify_loss_clusters(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if frame.empty:
            return rows

        for window_label, window_group in frame.groupby("window_label", sort=True):
            ordered = window_group.sort_values("signal_time").reset_index(drop=True)
            cluster: list[dict[str, Any]] = []
            cluster_id = 0
            for _, row in ordered.iterrows():
                record = row.to_dict()
                if float(record["pnl_r"]) < 0:
                    cluster.append(record)
                    continue
                if cluster:
                    cluster_id += 1
                    rows.append(self._build_cluster_row(window_label=window_label, cluster_id=cluster_id, items=cluster))
                    cluster = []
            if cluster:
                cluster_id += 1
                rows.append(self._build_cluster_row(window_label=window_label, cluster_id=cluster_id, items=cluster))
        return rows

    def summarize_loss_clusters(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        summary: list[dict[str, Any]] = []
        for window_label, group in frame.groupby("window_label", sort=True):
            worst = group.sort_values("cluster_cumulative_r").iloc[0]
            longest = group.sort_values(["cluster_length", "cluster_cumulative_r"], ascending=[False, True]).iloc[0]
            summary.append(
                {
                    "window_label": window_label,
                    "clusters": int(len(group)),
                    "loss_trades_in_clusters": int(group["cluster_length"].sum()),
                    "avg_cluster_length": round(float(group["cluster_length"].mean()), 2),
                    "median_cluster_length": round(float(group["cluster_length"].median()), 2),
                    "max_cluster_length": int(group["cluster_length"].max()),
                    "avg_cluster_cumulative_r": round(float(group["cluster_cumulative_r"].mean()), 4),
                    "worst_cluster_cumulative_r": round(float(worst["cluster_cumulative_r"]), 4),
                    "worst_cluster_length": int(worst["cluster_length"]),
                    "worst_cluster_start": str(worst["cluster_start"]),
                    "worst_cluster_end": str(worst["cluster_end"]),
                    "longest_cluster_length": int(longest["cluster_length"]),
                    "longest_cluster_cumulative_r": round(float(longest["cluster_cumulative_r"]), 4),
                    "longest_cluster_start": str(longest["cluster_start"]),
                    "longest_cluster_end": str(longest["cluster_end"]),
                }
            )
        return summary

    @staticmethod
    def _build_cluster_row(*, window_label: str, cluster_id: int, items: list[dict[str, Any]]) -> dict[str, Any]:
        frame = pd.DataFrame(items)
        return {
            "window_label": window_label,
            "cluster_id": cluster_id,
            "cluster_start": pd.Timestamp(frame["signal_time"].min()).isoformat(),
            "cluster_end": pd.Timestamp(frame["signal_time"].max()).isoformat(),
            "cluster_length": int(len(frame)),
            "cluster_cumulative_r": round(float(frame["pnl_r"].sum()), 4),
            "avg_loss_r": round(float(frame["pnl_r"].mean()), 4),
            "worst_trade_r": round(float(frame["pnl_r"].min()), 4),
            "avg_trend_strength": round(float(frame["trend_strength"].mean()), 2),
            "avg_bars_held": round(float(frame["bars_held"].mean()), 2),
            "avg_mfe_r": round(float(frame["mfe_r"].mean()), 4),
            "avg_mae_r": round(float(frame["mae_r"].mean()), 4),
            "long_count": int((frame["side"] == "LONG").sum()),
            "short_count": int((frame["side"] == "SHORT").sum()),
            "years": ",".join(str(int(item)) for item in sorted(frame["year"].unique())),
        }
