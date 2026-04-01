from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


HORIZONS = (1, 4, 24)


@dataclass(frozen=True)
class StateFeatureSpec:
    label: str
    value_col: str
    bucket_source_col: str


FEATURE_SPECS = [
    StateFeatureSpec(
        label="funding_zscore",
        value_col="funding_rate",
        bucket_source_col="funding_rate_z_7d",
    ),
    StateFeatureSpec(
        label="oi_change",
        value_col="open_interest_change_1h_pct",
        bucket_source_col="open_interest_change_z_7d",
    ),
    StateFeatureSpec(
        label="mark_index_premium",
        value_col="mark_index_spread_bps",
        bucket_source_col="mark_index_spread_bps_z_7d",
    ),
    StateFeatureSpec(
        label="basis",
        value_col="basis_proxy_bps",
        bucket_source_col="basis_proxy_bps_z_7d",
    ),
]


class DerivativesStateStudyService:
    def build_calibration_edges(self, calibration: pd.DataFrame) -> dict[str, list[float]]:
        edges: dict[str, list[float]] = {}
        for spec in FEATURE_SPECS:
            values = calibration[spec.bucket_source_col].dropna()
            if values.empty:
                raise ValueError(f"No calibration values for feature {spec.label}")
            edges[spec.label] = values.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        return edges

    def assign_feature_buckets(self, frame: pd.DataFrame, edges: dict[str, list[float]]) -> pd.DataFrame:
        enriched = frame.copy()
        for spec in FEATURE_SPECS:
            feature_edges = edges[spec.label]
            enriched[f"{spec.label}_bucket"] = pd.cut(
                enriched[spec.bucket_source_col],
                bins=[float("-inf"), feature_edges[0], feature_edges[1], feature_edges[2], feature_edges[3], float("inf")],
                labels=[1, 2, 3, 4, 5],
                include_lowest=True,
            ).astype("float64")
        return enriched

    def summarize_return_conditionality(
        self,
        frame: pd.DataFrame,
        *,
        min_bucket_obs: int = 100,
        min_monthly_bucket_obs: int = 8,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        bucket_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        monthly_rows: list[dict[str, Any]] = []

        if frame.empty:
            return bucket_rows, edge_rows, monthly_rows

        enriched = frame.copy()
        enriched["month"] = pd.to_datetime(enriched["timestamp"], utc=True).dt.strftime("%Y-%m")

        for spec in FEATURE_SPECS:
            bucket_col = f"{spec.label}_bucket"
            for horizon in HORIZONS:
                forward_col = f"forward_index_return_bps_{horizon}h"
                subset = enriched[[bucket_col, spec.value_col, "month", forward_col]].dropna().copy()
                if subset.empty:
                    continue

                grouped = []
                for bucket, group in subset.groupby(bucket_col, sort=True):
                    if len(group) < min_bucket_obs:
                        continue
                    grouped.append(
                        {
                            "feature": spec.label,
                            "horizon_hours": horizon,
                            "bucket": int(bucket),
                            "observations": int(len(group)),
                            "feature_mean": round(float(group[spec.value_col].mean()), 4),
                            "mean_forward_bps": round(float(group[forward_col].mean()), 4),
                            "median_forward_bps": round(float(group[forward_col].median()), 4),
                            "up_rate_pct": round(float((group[forward_col] > 0).mean() * 100.0), 2),
                        }
                    )
                bucket_rows.extend(grouped)

                grouped_frame = pd.DataFrame(grouped)
                if {1, 3, 5}.issubset(set(grouped_frame["bucket"])) if not grouped_frame.empty else False:
                    q1 = grouped_frame.loc[grouped_frame["bucket"] == 1].iloc[0]
                    q3 = grouped_frame.loc[grouped_frame["bucket"] == 3].iloc[0]
                    q5 = grouped_frame.loc[grouped_frame["bucket"] == 5].iloc[0]
                    overall_diff = float(q5["mean_forward_bps"] - q1["mean_forward_bps"])

                    month_pivots = (
                        subset.groupby(["month", bucket_col])[forward_col]
                        .agg(["mean", "size"])
                        .reset_index()
                        .pivot(index="month", columns=bucket_col, values=["mean", "size"])
                    )
                    stable_months = 0
                    agreeing_months = 0
                    for month in month_pivots.index:
                        q1_count = month_pivots.loc[month, ("size", 1.0)] if ("size", 1.0) in month_pivots.columns else None
                        q5_count = month_pivots.loc[month, ("size", 5.0)] if ("size", 5.0) in month_pivots.columns else None
                        if pd.isna(q1_count) or pd.isna(q5_count):
                            continue
                        if int(q1_count) < min_monthly_bucket_obs or int(q5_count) < min_monthly_bucket_obs:
                            continue
                        month_diff = float(month_pivots.loc[month, ("mean", 5.0)] - month_pivots.loc[month, ("mean", 1.0)])
                        stable_months += 1
                        if overall_diff == 0.0 or month_diff == 0.0:
                            agreeing_months += 1
                        elif overall_diff > 0 and month_diff > 0:
                            agreeing_months += 1
                        elif overall_diff < 0 and month_diff < 0:
                            agreeing_months += 1
                        monthly_rows.append(
                            {
                                "feature": spec.label,
                                "horizon_hours": horizon,
                                "month": month,
                                "q5_minus_q1_forward_bps": round(month_diff, 4),
                                "q1_observations": int(q1_count),
                                "q5_observations": int(q5_count),
                            }
                        )

                    edge_rows.append(
                        {
                            "feature": spec.label,
                            "horizon_hours": horizon,
                            "q5_minus_q1_forward_bps": round(overall_diff, 4),
                            "q5_minus_q3_forward_bps": round(float(q5["mean_forward_bps"] - q3["mean_forward_bps"]), 4),
                            "q1_minus_q3_forward_bps": round(float(q1["mean_forward_bps"] - q3["mean_forward_bps"]), 4),
                            "q5_minus_q1_up_rate_pct": round(float(q5["up_rate_pct"] - q1["up_rate_pct"]), 2),
                            "stable_months": stable_months,
                            "same_sign_months": agreeing_months,
                            "same_sign_month_rate_pct": round((agreeing_months / stable_months) * 100.0, 2) if stable_months else 0.0,
                        }
                    )

        return bucket_rows, edge_rows, monthly_rows

    def summarize_volatility_conditionality(
        self,
        frame: pd.DataFrame,
        *,
        min_bucket_obs: int = 100,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        bucket_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        if frame.empty:
            return bucket_rows, edge_rows

        for spec in FEATURE_SPECS:
            bucket_col = f"{spec.label}_bucket"
            for horizon in HORIZONS:
                abs_col = f"forward_abs_index_return_bps_{horizon}h"
                subset = frame[[bucket_col, spec.value_col, abs_col]].dropna().copy()
                if subset.empty:
                    continue

                grouped = []
                for bucket, group in subset.groupby(bucket_col, sort=True):
                    if len(group) < min_bucket_obs:
                        continue
                    grouped.append(
                        {
                            "feature": spec.label,
                            "horizon_hours": horizon,
                            "bucket": int(bucket),
                            "observations": int(len(group)),
                            "feature_mean": round(float(group[spec.value_col].mean()), 4),
                            "mean_abs_forward_bps": round(float(group[abs_col].mean()), 4),
                            "median_abs_forward_bps": round(float(group[abs_col].median()), 4),
                        }
                    )
                bucket_rows.extend(grouped)

                grouped_frame = pd.DataFrame(grouped)
                if {1, 3, 5}.issubset(set(grouped_frame["bucket"])) if not grouped_frame.empty else False:
                    q1 = grouped_frame.loc[grouped_frame["bucket"] == 1].iloc[0]
                    q3 = grouped_frame.loc[grouped_frame["bucket"] == 3].iloc[0]
                    q5 = grouped_frame.loc[grouped_frame["bucket"] == 5].iloc[0]
                    extreme = max(float(q1["mean_abs_forward_bps"]), float(q5["mean_abs_forward_bps"]))
                    edge_rows.append(
                        {
                            "feature": spec.label,
                            "horizon_hours": horizon,
                            "q5_minus_q1_abs_forward_bps": round(float(q5["mean_abs_forward_bps"] - q1["mean_abs_forward_bps"]), 4),
                            "q1_to_q3_abs_vol_ratio": round(float(q1["mean_abs_forward_bps"] / q3["mean_abs_forward_bps"]), 4) if q3["mean_abs_forward_bps"] else None,
                            "q5_to_q3_abs_vol_ratio": round(float(q5["mean_abs_forward_bps"] / q3["mean_abs_forward_bps"]), 4) if q3["mean_abs_forward_bps"] else None,
                            "extreme_to_q3_abs_vol_ratio": round(float(extreme / q3["mean_abs_forward_bps"]), 4) if q3["mean_abs_forward_bps"] else None,
                        }
                    )

        return bucket_rows, edge_rows

    def summarize_mainline_trade_state(
        self,
        trades: pd.DataFrame,
        *,
        min_bucket_obs: int = 5,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        bucket_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        exit_rows: list[dict[str, Any]] = []
        if trades.empty:
            return bucket_rows, edge_rows, exit_rows

        for side_label, side_frame in [
            ("ALL", trades),
            ("LONG", trades[trades["side"] == "LONG"]),
            ("SHORT", trades[trades["side"] == "SHORT"]),
        ]:
            baseline = self._summarize_trade_subset(side_frame)
            for spec in FEATURE_SPECS:
                bucket_col = f"{spec.label}_bucket"
                feature_frame = side_frame.dropna(subset=[bucket_col]).copy()
                grouped = []
                for bucket, group in feature_frame.groupby(bucket_col, sort=True):
                    if len(group) < min_bucket_obs:
                        continue
                    summary = self._summarize_trade_subset(group)
                    grouped.append(
                        {
                            "feature": spec.label,
                            "side": side_label,
                            "bucket": int(bucket),
                            **summary,
                        }
                    )
                    exit_counts = (
                        group["exit_reason"].value_counts(normalize=True).sort_values(ascending=False).head(3)
                    )
                    for exit_reason, rate in exit_counts.items():
                        exit_rows.append(
                            {
                                "feature": spec.label,
                                "side": side_label,
                                "bucket": int(bucket),
                                "exit_reason": exit_reason,
                                "rate_pct": round(float(rate * 100.0), 2),
                            }
                        )

                bucket_rows.extend(grouped)
                grouped_frame = pd.DataFrame(grouped)
                if {1, 5}.issubset(set(grouped_frame["bucket"])) if not grouped_frame.empty else False:
                    q1 = grouped_frame.loc[grouped_frame["bucket"] == 1].iloc[0]
                    q5 = grouped_frame.loc[grouped_frame["bucket"] == 5].iloc[0]
                    edge_rows.append(
                        {
                            "feature": spec.label,
                            "side": side_label,
                            "baseline_trades": baseline["trades"],
                            "baseline_win_rate_pct": baseline["win_rate_pct"],
                            "baseline_expectancy_r": baseline["expectancy_r"],
                            "baseline_max_dd_r": baseline["max_drawdown_r"],
                            "baseline_avg_hold_bars": baseline["avg_bars_held"],
                            "q5_minus_q1_win_rate_pct": round(float(q5["win_rate_pct"] - q1["win_rate_pct"]), 2),
                            "q5_minus_q1_expectancy_r": round(float(q5["expectancy_r"] - q1["expectancy_r"]), 4),
                            "q5_minus_q1_max_dd_r": round(float(q5["max_drawdown_r"] - q1["max_drawdown_r"]), 4),
                            "q5_minus_q1_avg_hold_bars": round(float(q5["avg_bars_held"] - q1["avg_bars_held"]), 2),
                            "q1_minus_baseline_expectancy_r": round(float(q1["expectancy_r"] - baseline["expectancy_r"]), 4),
                            "q5_minus_baseline_expectancy_r": round(float(q5["expectancy_r"] - baseline["expectancy_r"]), 4),
                            "q1_minus_baseline_win_rate_pct": round(float(q1["win_rate_pct"] - baseline["win_rate_pct"]), 2),
                            "q5_minus_baseline_win_rate_pct": round(float(q5["win_rate_pct"] - baseline["win_rate_pct"]), 2),
                        }
                    )

        return bucket_rows, edge_rows, exit_rows

    def _summarize_trade_subset(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {
                "trades": 0,
                "win_rate_pct": 0.0,
                "expectancy_r": 0.0,
                "cum_r": 0.0,
                "max_drawdown_r": 0.0,
                "avg_bars_held": 0.0,
                "median_bars_held": 0.0,
                "tp1_hit_rate_pct": 0.0,
                "tp2_hit_rate_pct": 0.0,
            }

        ordered = frame.sort_values("signal_time").reset_index(drop=True)
        cumulative = ordered["pnl_r"].cumsum()
        drawdown = cumulative - cumulative.cummax()
        return {
            "trades": int(len(ordered)),
            "win_rate_pct": round(float((ordered["pnl_r"] > 0).mean() * 100.0), 2),
            "expectancy_r": round(float(ordered["pnl_r"].mean()), 4),
            "cum_r": round(float(ordered["pnl_r"].sum()), 4),
            "max_drawdown_r": round(float(drawdown.min()), 4),
            "avg_bars_held": round(float(ordered["bars_held"].mean()), 2),
            "median_bars_held": round(float(ordered["bars_held"].median()), 2),
            "tp1_hit_rate_pct": round(float(ordered["tp1_hit"].mean() * 100.0), 2),
            "tp2_hit_rate_pct": round(float(ordered["tp2_hit"].mean() * 100.0), 2),
        }
