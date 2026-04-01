from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


PRIMARY_HORIZONS = (24, 72, 168)
ROUNDTRIP_COSTS_BPS = (0.0, 10.0, 20.0, 28.0)


@dataclass(frozen=True)
class CarryCandidate:
    label: str
    description: str
    selector: Callable[[pd.DataFrame], pd.Series]


def _future_sum_exclusive(series: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    values = series.fillna(0.0).to_numpy(dtype=float)
    length = len(values)
    result = np.full(length, np.nan)
    valid_count = length - horizon
    if valid_count <= 0:
        return pd.Series(result, index=series.index, dtype="float64")

    cumulative = np.cumsum(np.concatenate([[0.0], values]))
    idx = np.arange(valid_count)
    result[idx] = cumulative[idx + horizon + 1] - cumulative[idx + 1]
    return pd.Series(result, index=series.index, dtype="float64")


class CarryBasisResearchService:
    def build_calibration_edges(self, calibration: pd.DataFrame) -> dict[str, list[float]]:
        edges: dict[str, list[float]] = {}
        for label, column in {
            "funding_zscore": "funding_rate_z_7d",
            "basis_zscore": "basis_proxy_bps_z_7d",
        }.items():
            values = calibration[column].dropna()
            if values.empty:
                raise ValueError(f"No calibration values for {label}")
            edges[label] = values.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        return edges

    def enrich_frame(self, frame: pd.DataFrame, edges: dict[str, list[float]]) -> pd.DataFrame:
        enriched = frame.copy().sort_values("timestamp").reset_index(drop=True)
        enriched["funding_event_bps"] = enriched["funding_rate_event"].fillna(0.0) * 10000.0
        enriched["basis_positive"] = enriched["basis_proxy_bps"] > 0.0
        enriched["funding_positive"] = enriched["funding_rate"] > 0.0
        enriched["funding_zscore_bucket"] = pd.cut(
            enriched["funding_rate_z_7d"],
            bins=[float("-inf"), *edges["funding_zscore"], float("inf")],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True,
        ).astype("float64")
        enriched["basis_zscore_bucket"] = pd.cut(
            enriched["basis_proxy_bps_z_7d"],
            bins=[float("-inf"), *edges["basis_zscore"], float("inf")],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True,
        ).astype("float64")

        for horizon in PRIMARY_HORIZONS:
            future_index = enriched["index_close"].shift(-horizon)
            future_mark = enriched["mark_close"].shift(-horizon)
            spot_leg_bps = ((future_index / enriched["index_close"]) - 1.0) * 10000.0
            perp_leg_short_bps = -((future_mark / enriched["mark_close"]) - 1.0) * 10000.0
            funding_sum_bps = _future_sum_exclusive(enriched["funding_event_bps"], horizon=horizon)

            enriched[f"gross_carry_bps_{horizon}h"] = spot_leg_bps + perp_leg_short_bps + funding_sum_bps
            enriched[f"gross_reverse_carry_bps_{horizon}h"] = -(spot_leg_bps + perp_leg_short_bps + funding_sum_bps)
            for cost in ROUNDTRIP_COSTS_BPS:
                suffix = self.cost_suffix(cost)
                enriched[f"net_carry_bps_{horizon}h_{suffix}"] = enriched[f"gross_carry_bps_{horizon}h"] - cost
                enriched[f"net_reverse_carry_bps_{horizon}h_{suffix}"] = enriched[f"gross_reverse_carry_bps_{horizon}h"] - cost

        return enriched

    def summarize_unconditional(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for horizon in PRIMARY_HORIZONS:
            gross_col = f"gross_carry_bps_{horizon}h"
            reverse_col = f"gross_reverse_carry_bps_{horizon}h"
            for label, column in [
                ("long_spot_short_perp", gross_col),
                ("short_spot_long_perp_diagnostic", reverse_col),
            ]:
                subset = frame[[column]].dropna()
                if subset.empty:
                    continue
                row = {
                    "leg": label,
                    "horizon_hours": horizon,
                    "observations": int(len(subset)),
                    "gross_mean_bps": round(float(subset[column].mean()), 4),
                    "gross_median_bps": round(float(subset[column].median()), 4),
                    "gross_positive_rate_pct": round(float((subset[column] > 0).mean() * 100.0), 2),
                }
                for cost in ROUNDTRIP_COSTS_BPS:
                    suffix = self.cost_suffix(cost)
                    net_values = subset[column] - cost
                    row[f"net_mean_bps_{suffix}"] = round(float(net_values.mean()), 4)
                    row[f"net_positive_rate_pct_{suffix}"] = round(float((net_values > 0).mean() * 100.0), 2)
                rows.append(row)
        return rows

    def build_candidates(self) -> list[CarryCandidate]:
        return [
            CarryCandidate(
                label="always_on",
                description="No state filter; open carry at every eligible hour.",
                selector=lambda frame: pd.Series(True, index=frame.index),
            ),
            CarryCandidate(
                label="basis_positive",
                description="Only when raw basis proxy is positive.",
                selector=lambda frame: frame["basis_positive"],
            ),
            CarryCandidate(
                label="funding_positive",
                description="Only when current funding rate is positive.",
                selector=lambda frame: frame["funding_positive"],
            ),
            CarryCandidate(
                label="basis_and_funding_positive",
                description="Only when both raw basis and funding are positive.",
                selector=lambda frame: frame["basis_positive"] & frame["funding_positive"],
            ),
            CarryCandidate(
                label="basis_q5",
                description="Only when basis z-score bucket is Q5.",
                selector=lambda frame: frame["basis_zscore_bucket"] == 5.0,
            ),
            CarryCandidate(
                label="funding_q5",
                description="Only when funding z-score bucket is Q5.",
                selector=lambda frame: frame["funding_zscore_bucket"] == 5.0,
            ),
            CarryCandidate(
                label="basis_q5_and_funding_q5",
                description="Only when both basis and funding z-score buckets are Q5.",
                selector=lambda frame: (frame["basis_zscore_bucket"] == 5.0) & (frame["funding_zscore_bucket"] == 5.0),
            ),
        ]

    def summarize_candidates(self, frame: pd.DataFrame, candidates: list[CarryCandidate]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            mask = candidate.selector(frame)
            subset_base = frame[mask].copy()
            for horizon in PRIMARY_HORIZONS:
                subset = subset_base[[f"gross_carry_bps_{horizon}h"]].dropna()
                if subset.empty:
                    continue
                row = {
                    "candidate": candidate.label,
                    "description": candidate.description,
                    "horizon_hours": horizon,
                    "observations": int(len(subset)),
                    "gross_mean_bps": round(float(subset[f"gross_carry_bps_{horizon}h"].mean()), 4),
                    "gross_median_bps": round(float(subset[f"gross_carry_bps_{horizon}h"].median()), 4),
                    "gross_positive_rate_pct": round(float((subset[f"gross_carry_bps_{horizon}h"] > 0).mean() * 100.0), 2),
                }
                for cost in ROUNDTRIP_COSTS_BPS:
                    suffix = self.cost_suffix(cost)
                    values = subset[f"gross_carry_bps_{horizon}h"] - cost
                    row[f"net_mean_bps_{suffix}"] = round(float(values.mean()), 4)
                    row[f"net_median_bps_{suffix}"] = round(float(values.median()), 4)
                    row[f"net_positive_rate_pct_{suffix}"] = round(float((values > 0).mean() * 100.0), 2)
                rows.append(row)
        return rows

    def summarize_monthly_stability(
        self,
        frame: pd.DataFrame,
        *,
        candidate: CarryCandidate,
        horizon: int,
        cost_bps: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        mask = candidate.selector(frame)
        suffix = self.cost_suffix(cost_bps)
        column = f"net_carry_bps_{horizon}h_{suffix}"
        subset = frame.loc[mask, ["timestamp", column]].dropna().copy()
        subset["month"] = pd.to_datetime(subset["timestamp"], utc=True).dt.strftime("%Y-%m")
        grouped = subset.groupby("month", sort=True)[column].agg(["size", "mean", "median"]).reset_index()
        rows = [
            {
                "candidate": candidate.label,
                "horizon_hours": horizon,
                "cost_bps": cost_bps,
                "month": row["month"],
                "observations": int(row["size"]),
                "net_mean_bps": round(float(row["mean"]), 4),
                "net_median_bps": round(float(row["median"]), 4),
            }
            for _, row in grouped.iterrows()
        ]
        summary = {
            "candidate": candidate.label,
            "horizon_hours": horizon,
            "cost_bps": cost_bps,
            "months": int(len(grouped)),
            "positive_months": int((grouped["mean"] > 0).sum()) if not grouped.empty else 0,
            "positive_month_rate_pct": round(float((grouped["mean"] > 0).mean() * 100.0), 2) if not grouped.empty else 0.0,
        }
        return rows, summary

    def choose_best_candidate(self, candidate_rows: list[dict[str, Any]], *, cost_bps: float = 28.0) -> dict[str, Any] | None:
        if not candidate_rows:
            return None
        suffix = self.cost_suffix(cost_bps)
        frame = pd.DataFrame(candidate_rows)
        frame = frame[frame["horizon_hours"].isin([72, 168])].copy()
        frame["selection_score"] = frame[f"net_mean_bps_{suffix}"] + frame[f"net_positive_rate_pct_{suffix}"] / 100.0
        if frame.empty:
            return None
        return frame.sort_values(
            ["selection_score", f"net_mean_bps_{suffix}", "observations"],
            ascending=False,
        ).iloc[0].to_dict()

    @staticmethod
    def cost_suffix(cost_bps: float) -> str:
        if float(cost_bps).is_integer():
            return f"cost_{int(cost_bps)}bps"
        return f"cost_{str(cost_bps).replace('.', '_')}bps"
