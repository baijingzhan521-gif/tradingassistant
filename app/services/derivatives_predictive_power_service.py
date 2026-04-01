from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


STATE_LOW_THRESHOLD = -1.0
STATE_HIGH_THRESHOLD = 1.0
HORIZONS = (1, 4, 24)


@dataclass(frozen=True)
class FeatureSpec:
    label: str
    value_col: str | None
    state_col: str | None
    available: bool = True
    note: str | None = None


FEATURE_SPECS = [
    FeatureSpec(
        label="funding_zscore",
        value_col="funding_rate_z_7d",
        state_col="funding_rate_z_7d",
    ),
    FeatureSpec(
        label="oi_change",
        value_col="open_interest_change_1h_pct",
        state_col="open_interest_change_z_7d",
    ),
    FeatureSpec(
        label="mark_index_premium",
        value_col="mark_index_spread_bps",
        state_col="mark_index_spread_bps_z_7d",
    ),
    FeatureSpec(
        label="basis",
        value_col="basis_proxy_bps",
        state_col="basis_proxy_bps_z_7d",
    ),
    FeatureSpec(
        label="liquidation_burst",
        value_col=None,
        state_col=None,
        available=False,
        note="Official historical backfill is unavailable in the current data pipeline.",
    ),
]


class DerivativesPredictivePowerService:
    def summarize_availability(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in FEATURE_SPECS:
            if not spec.available:
                rows.append(
                    {
                        "feature": spec.label,
                        "available": False,
                        "rows": 0,
                        "note": spec.note or "",
                    }
                )
                continue

            usable = frame[[spec.value_col, spec.state_col]].dropna() if spec.value_col and spec.state_col else pd.DataFrame()
            rows.append(
                {
                    "feature": spec.label,
                    "available": True,
                    "rows": int(len(usable)),
                    "note": spec.note or "",
                }
            )
        return rows

    def summarize_correlations(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in FEATURE_SPECS:
            if not spec.available or spec.value_col is None:
                continue
            for horizon in HORIZONS:
                forward_col = f"forward_index_return_bps_{horizon}h"
                abs_forward_col = f"forward_abs_index_return_bps_{horizon}h"
                subset = frame[[spec.value_col, forward_col, abs_forward_col]].dropna()
                if subset.empty:
                    continue
                rows.append(
                    {
                        "feature": spec.label,
                        "horizon_hours": horizon,
                        "observations": int(len(subset)),
                        "corr_forward_bps": round(float(subset[spec.value_col].corr(subset[forward_col])), 4),
                        "corr_abs_forward_bps": round(float(subset[spec.value_col].corr(subset[abs_forward_col])), 4),
                    }
                )
        return rows

    def summarize_state_tables(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in FEATURE_SPECS:
            if not spec.available or spec.value_col is None or spec.state_col is None:
                continue
            selected_columns = list(dict.fromkeys([spec.value_col, spec.state_col] + self._forward_columns()))
            feature_frame = frame[selected_columns].dropna().copy()
            if feature_frame.empty:
                continue
            feature_frame["state"] = feature_frame[spec.state_col].map(self._state_label)
            for horizon in HORIZONS:
                forward_col = f"forward_index_return_bps_{horizon}h"
                abs_forward_col = f"forward_abs_index_return_bps_{horizon}h"
                for state, group in feature_frame.groupby("state", sort=False):
                    if group.empty:
                        continue
                    rows.append(
                        {
                            "feature": spec.label,
                            "horizon_hours": horizon,
                            "state": state,
                            "observations": int(len(group)),
                            "feature_mean": round(float(group[spec.value_col].mean()), 4),
                            "mean_forward_bps": round(float(group[forward_col].mean()), 4),
                            "median_forward_bps": round(float(group[forward_col].median()), 4),
                            "up_rate_pct": round(float((group[forward_col] > 0).mean() * 100.0), 2),
                            "mean_abs_forward_bps": round(float(group[abs_forward_col].mean()), 4),
                        }
                    )
        return rows

    def summarize_feature_edges(self, state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not state_rows:
            return []
        frame = pd.DataFrame(state_rows)
        rows: list[dict[str, Any]] = []
        for (feature, horizon), group in frame.groupby(["feature", "horizon_hours"], sort=True):
            low = group.loc[group["state"] == "low"]
            mid = group.loc[group["state"] == "mid"]
            high = group.loc[group["state"] == "high"]
            if low.empty or mid.empty or high.empty:
                continue
            low_row = low.iloc[0]
            mid_row = mid.iloc[0]
            high_row = high.iloc[0]
            mid_abs = float(mid_row["mean_abs_forward_bps"])
            max_extreme_abs = max(float(low_row["mean_abs_forward_bps"]), float(high_row["mean_abs_forward_bps"]))
            rows.append(
                {
                    "feature": feature,
                    "horizon_hours": int(horizon),
                    "high_minus_low_forward_bps": round(float(high_row["mean_forward_bps"] - low_row["mean_forward_bps"]), 4),
                    "high_minus_mid_forward_bps": round(float(high_row["mean_forward_bps"] - mid_row["mean_forward_bps"]), 4),
                    "low_minus_mid_forward_bps": round(float(low_row["mean_forward_bps"] - mid_row["mean_forward_bps"]), 4),
                    "extreme_to_mid_abs_vol_ratio": round((max_extreme_abs / mid_abs), 4) if mid_abs else None,
                }
            )
        return rows

    @staticmethod
    def _state_label(value: float) -> str:
        if value <= STATE_LOW_THRESHOLD:
            return "low"
        if value >= STATE_HIGH_THRESHOLD:
            return "high"
        return "mid"

    @staticmethod
    def _forward_columns() -> list[str]:
        columns: list[str] = []
        for horizon in HORIZONS:
            columns.extend(
                [
                    f"forward_index_return_bps_{horizon}h",
                    f"forward_abs_index_return_bps_{horizon}h",
                ]
            )
        return columns
