from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas.common import Action


REQUIRED_COLUMNS = ("timestamp", "permission_label", "allow_long", "allow_short", "model_version")
ALLOWED_PERMISSION_LABELS = {"allow_long", "allow_short", "allow_none"}


@dataclass(frozen=True)
class SidePermissionState:
    timestamp: pd.Timestamp
    permission_label: str
    allow_long: bool
    allow_short: bool
    model_version: str
    long_score: float | None = None
    short_score: float | None = None
    meta_regime: str | None = None


class SidePermissionResearchService:
    def load_permission_csv(self, path: str | Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Permission CSV missing required columns: {', '.join(missing)}")

        enriched = frame.copy()
        enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True)
        if enriched["timestamp"].isna().any():
            raise ValueError("Permission CSV contains invalid timestamps")

        enriched["allow_long"] = self._coerce_bool_series(enriched["allow_long"], "allow_long")
        enriched["allow_short"] = self._coerce_bool_series(enriched["allow_short"], "allow_short")
        enriched["permission_label"] = enriched["permission_label"].astype("string")
        enriched["model_version"] = enriched["model_version"].astype("string")
        if enriched["model_version"].isna().any() or (enriched["model_version"].str.strip() == "").any():
            raise ValueError("Permission CSV contains empty model_version")

        invalid_labels = sorted(set(enriched["permission_label"].dropna()) - ALLOWED_PERMISSION_LABELS)
        if invalid_labels:
            raise ValueError(f"Permission CSV contains invalid permission_label values: {', '.join(invalid_labels)}")
        if enriched["permission_label"].isna().any():
            raise ValueError("Permission CSV contains missing permission_label")

        if enriched["timestamp"].duplicated().any():
            duplicates = (
                enriched.loc[enriched["timestamp"].duplicated(keep=False), "timestamp"]
                .astype("string")
                .tolist()
            )
            raise ValueError(f"Permission CSV contains duplicate timestamps: {', '.join(sorted(set(duplicates))[:5])}")

        expected_labels = enriched.apply(self._expected_label, axis=1)
        mismatched = enriched.loc[expected_labels != enriched["permission_label"]]
        if not mismatched.empty:
            sample = mismatched.iloc[0]
            raise ValueError(
                "permission_label is inconsistent with allow_long/allow_short at "
                f"{pd.Timestamp(sample['timestamp']).isoformat()}"
            )

        optional_columns = ("long_score", "short_score", "meta_regime")
        for column in optional_columns:
            if column not in enriched.columns:
                enriched[column] = None

        return enriched.sort_values("timestamp").reset_index(drop=True)

    def resolve_permission(self, permissions: pd.DataFrame, timestamp: pd.Timestamp | str) -> SidePermissionState | None:
        if permissions.empty:
            return None
        point = pd.Timestamp(timestamp)
        point = point.tz_localize("UTC") if point.tzinfo is None else point.tz_convert("UTC")
        idx = int(permissions["timestamp"].searchsorted(point, side="right") - 1)
        if idx < 0:
            return None
        row = permissions.iloc[idx]
        return SidePermissionState(
            timestamp=pd.Timestamp(row["timestamp"]),
            permission_label=str(row["permission_label"]),
            allow_long=bool(row["allow_long"]),
            allow_short=bool(row["allow_short"]),
            model_version=str(row["model_version"]),
            long_score=self._maybe_float(row.get("long_score")),
            short_score=self._maybe_float(row.get("short_score")),
            meta_regime=None if pd.isna(row.get("meta_regime")) else str(row.get("meta_regime")),
        )

    def should_veto(
        self,
        *,
        variant: str,
        action: Action,
        permission: SidePermissionState | None,
    ) -> bool:
        if variant == "baseline_no_permission" or permission is None:
            return False
        if action == Action.LONG:
            if variant == "permission_long_only":
                return not permission.allow_long
            if variant == "permission_full_side_control":
                return not permission.allow_long
            return False
        if action == Action.SHORT:
            if variant == "permission_short_only":
                return not permission.allow_short
            if variant == "permission_full_side_control":
                return not permission.allow_short
            return False
        return False

    def summarize_trade_distribution(self, frame: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, Any]]:
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
                    "avg_bars_held": round(float(group["bars_held"].mean()), 2),
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

    def identify_loss_clusters(self, frame: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if frame.empty:
            return rows

        grouped = frame.groupby(group_cols, sort=True, dropna=False, observed=True)
        for keys, group in grouped:
            keys = (keys,) if not isinstance(keys, tuple) else keys
            base = {column: value for column, value in zip(group_cols, keys)}
            ordered = group.sort_values("signal_time").reset_index(drop=True)
            cluster: list[dict[str, Any]] = []
            cluster_id = 0
            for _, row in ordered.iterrows():
                record = row.to_dict()
                if float(record["pnl_r"]) < 0:
                    cluster.append(record)
                    continue
                if cluster:
                    cluster_id += 1
                    rows.append(self._build_cluster_row(base=base, cluster_id=cluster_id, items=cluster))
                    cluster = []
            if cluster:
                cluster_id += 1
                rows.append(self._build_cluster_row(base=base, cluster_id=cluster_id, items=cluster))
        return rows

    def summarize_loss_clusters(self, rows: list[dict[str, Any]], *, group_cols: list[str]) -> list[dict[str, Any]]:
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        summary: list[dict[str, Any]] = []
        grouped = frame.groupby(group_cols, sort=True, dropna=False, observed=True)
        for keys, group in grouped:
            keys = (keys,) if not isinstance(keys, tuple) else keys
            row = {column: value for column, value in zip(group_cols, keys)}
            worst = group.sort_values("cluster_cumulative_r").iloc[0]
            longest = group.sort_values(["cluster_length", "cluster_cumulative_r"], ascending=[False, True]).iloc[0]
            row.update(
                {
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
            summary.append(row)
        return summary

    def summarize_gate_coverage(self, frame: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if frame.empty:
            return rows
        grouped = frame.groupby(group_cols, sort=True, dropna=False, observed=True)
        for keys, group in grouped:
            keys = (keys,) if not isinstance(keys, tuple) else keys
            row = {column: value for column, value in zip(group_cols, keys)}
            signals_now = int(len(group))
            covered = int(group["permission_covered"].sum())
            vetoed = int(group["vetoed"].sum())
            row.update(
                {
                    "signals_now": signals_now,
                    "signals_with_state": covered,
                    "gate_coverage_pct": round((covered / signals_now) * 100.0, 2) if signals_now else 0.0,
                    "vetoed_signals": vetoed,
                    "veto_rate_pct": round((vetoed / signals_now) * 100.0, 2) if signals_now else 0.0,
                }
            )
            rows.append(row)
        return rows

    @staticmethod
    def _build_cluster_row(*, base: dict[str, Any], cluster_id: int, items: list[dict[str, Any]]) -> dict[str, Any]:
        frame = pd.DataFrame(items)
        row = dict(base)
        row.update(
            {
                "cluster_id": cluster_id,
                "cluster_start": pd.Timestamp(frame["signal_time"].min()).isoformat(),
                "cluster_end": pd.Timestamp(frame["signal_time"].max()).isoformat(),
                "cluster_length": int(len(frame)),
                "cluster_cumulative_r": round(float(frame["pnl_r"].sum()), 4),
                "avg_loss_r": round(float(frame["pnl_r"].mean()), 4),
                "worst_trade_r": round(float(frame["pnl_r"].min()), 4),
                "avg_trend_strength": round(float(frame["trend_strength"].mean()), 2),
                "avg_bars_held": round(float(frame["bars_held"].mean()), 2),
                "long_count": int((frame["side"] == "LONG").sum()),
                "short_count": int((frame["side"] == "SHORT").sum()),
                "years": ",".join(str(int(item)) for item in sorted(frame["year"].unique())),
            }
        )
        return row

    @staticmethod
    def _coerce_bool_series(series: pd.Series, column: str) -> pd.Series:
        mapping = {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            1: True,
            0: False,
        }
        values = series.map(mapping)
        if values.isna().any():
            raise ValueError(f"Permission CSV contains non-boolean values in {column}")
        return values.astype(bool)

    @staticmethod
    def _expected_label(row: pd.Series) -> str:
        if bool(row["allow_long"]) and not bool(row["allow_short"]):
            return "allow_long"
        if bool(row["allow_short"]) and not bool(row["allow_long"]):
            return "allow_short"
        if not bool(row["allow_long"]) and not bool(row["allow_short"]):
            return "allow_none"
        return "invalid"

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)
