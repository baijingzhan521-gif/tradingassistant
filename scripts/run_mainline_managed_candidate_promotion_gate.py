from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestTrade
from app.core.logging import configure_logging
from scripts.post_tp1_managed_replay import (
    PROFILE_SPEC_MAP,
    baseline_profile_label,
    build_service,
    precompute_extension_features,
    run_profile,
)
from scripts.stability_check import iter_calendar_windows, iter_rolling_windows, summarize_trade_slice


DEFAULT_CHAMPION = "swing_trend_long_regime_gate_v1"
DEFAULT_CHALLENGER = "swing_trend_long_regime_short_no_reversal_no_aux_v1"
DEFAULT_OVERLAY = "be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "mainline_managed_candidate_promotion_gate"
PRIMARY_WINDOW = "primary_2020_latest"
SECONDARY_WINDOW = "secondary_2024_latest"
PRIMARY_WINDOW_LABEL = "2020-01-01 -> latest"
SECONDARY_WINDOW_LABEL = "2024-03-19 -> latest"
DIFF_YEARS = (2022, 2024, 2026)
MATCH_TOLERANCE = pd.Timedelta(hours=12)
COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
}


@dataclass(frozen=True)
class MatchPair:
    champion_idx: int
    challenger_idx: int
    kind: str
    gap_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promotion gate for managed champion vs challenger candidates.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--champion-profile", default=DEFAULT_CHAMPION)
    parser.add_argument("--challenger-profile", default=DEFAULT_CHALLENGER)
    parser.add_argument("--overlay-profile", default=DEFAULT_OVERLAY)
    parser.add_argument("--primary-start", default="2020-01-01")
    parser.add_argument("--secondary-start", default="2024-03-19")
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--rolling-180-step-days", type=int, default=30)
    parser.add_argument("--rolling-365-step-days", type=int, default=60)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def resolve_end_from_history(enriched_history: dict[str, dict[str, pd.DataFrame]]) -> datetime:
    profile_ends: list[pd.Timestamp] = []
    for frames in enriched_history.values():
        profile_ends.append(min(pd.Timestamp(frame["timestamp"].max()) for frame in frames.values()))
    return min(profile_ends).to_pydatetime()


def candidate_role(profile: str, *, champion: str, challenger: str) -> str:
    if profile == champion:
        return "champion_managed"
    if profile == challenger:
        return "challenger_managed"
    return profile


def candidate_label(profile: str, *, champion: str, challenger: str) -> str:
    if profile == champion:
        return "Champion + Universal Overlay"
    if profile == challenger:
        return "Challenger + Universal Overlay"
    return profile


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([head, sep, *body])


def summarize_object(summary) -> dict[str, Any]:
    return {
        "trades": int(summary.total_trades),
        "win_rate_pct": round(float(summary.win_rate), 2),
        "profit_factor": round(float(summary.profit_factor), 4),
        "expectancy_r": round(float(summary.expectancy_r), 4),
        "cum_r": round(float(summary.cumulative_r), 4),
        "max_dd_r": round(float(summary.max_drawdown_r), 4),
        "avg_holding_bars": round(float(summary.avg_holding_bars), 2),
        "tp1_hit_rate_pct": round(float(summary.tp1_hit_rate), 2),
        "tp2_hit_rate_pct": round(float(summary.tp2_hit_rate), 2),
        "signals_now": int(summary.signals_now),
        "skipped_entries": int(summary.skipped_entries),
    }


def summary_row(
    *,
    cost_scenario: str,
    window: str,
    window_label: str,
    strategy_profile: str,
    champion: str,
    challenger: str,
    overlay_profile: str,
    summary,
) -> dict[str, Any]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "window_label": window_label,
        "strategy_profile": strategy_profile,
        "candidate_role": candidate_role(strategy_profile, champion=champion, challenger=challenger),
        "candidate_label": candidate_label(strategy_profile, champion=champion, challenger=challenger),
        "baseline_profile_label": baseline_profile_label(strategy_profile),
        "overlay_profile": overlay_profile,
        **summarize_object(summary),
    }


def trade_rows_from_trades(
    *,
    cost_scenario: str,
    window: str,
    window_label: str,
    strategy_profile: str,
    champion: str,
    challenger: str,
    overlay_profile: str,
    trades: list[BacktestTrade],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        row = asdict(trade)
        row["cost_scenario"] = cost_scenario
        row["window"] = window
        row["window_label"] = window_label
        row["strategy_profile"] = strategy_profile
        row["candidate_role"] = candidate_role(strategy_profile, champion=champion, challenger=challenger)
        row["candidate_label"] = candidate_label(strategy_profile, champion=champion, challenger=challenger)
        row["baseline_profile_label"] = baseline_profile_label(strategy_profile)
        row["overlay_profile"] = overlay_profile
        rows.append(row)
    return rows


def build_trade_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "cost_scenario",
                "window",
                "strategy_profile",
                "candidate_role",
                "candidate_label",
                "side",
                "signal_time",
                "entry_time",
                "exit_time",
                "bars_held",
                "exit_reason",
                "tp1_hit",
                "tp2_hit",
                "pnl_r",
            ]
        )
    frame = pd.DataFrame(rows)
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["year"] = frame["entry_time"].dt.year.astype(int)
    frame["month"] = frame["entry_time"].dt.strftime("%Y-%m")
    frame["quarter"] = frame["entry_time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    return frame.sort_values(["cost_scenario", "window", "entry_time"]).reset_index(drop=True)


def compute_group_max_drawdown(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_values("entry_time")
    cumulative = ordered["pnl_r"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    return round(abs(float(drawdown.min())), 4)


def summarize_trade_distribution(frame: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_cols, sort=True, dropna=False, observed=True)
    for keys, group in grouped:
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = {column: value for column, value in zip(group_cols, keys)}
        wins = float(group.loc[group["pnl_r"] > 0, "pnl_r"].sum())
        losses = float(-group.loc[group["pnl_r"] < 0, "pnl_r"].sum())
        profit_factor = round(wins / losses, 4) if losses > 0 else None
        row.update(
            {
                "trades": int(len(group)),
                "win_rate_pct": round(float((group["pnl_r"] > 0).mean() * 100.0), 2),
                "expectancy_r": round(float(group["pnl_r"].mean()), 4),
                "cum_r": round(float(group["pnl_r"].sum()), 4),
                "profit_factor": profit_factor,
                "max_dd_r": compute_group_max_drawdown(group),
                "avg_bars_held": round(float(group["bars_held"].mean()), 2),
            }
        )
        rows.append(row)
    return rows


def normalized_slice_row(
    *,
    raw_row: dict[str, Any],
    cost_scenario: str,
    window: str,
    strategy_profile: str,
    champion: str,
    challenger: str,
) -> dict[str, Any]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "strategy_profile": strategy_profile,
        "candidate_role": candidate_role(strategy_profile, champion=champion, challenger=challenger),
        "candidate_label": candidate_label(strategy_profile, champion=champion, challenger=challenger),
        "label": raw_row["label"],
        "horizon": raw_row["horizon"],
        "start": raw_row["start"],
        "end": raw_row["end"],
        "days": raw_row["days"],
        "trades": raw_row["trades"],
        "win_rate_pct": round(float(raw_row["win_rate"]), 2),
        "profit_factor": round(float(raw_row["profit_factor"]), 4),
        "expectancy_r": round(float(raw_row["expectancy_r"]), 4),
        "cum_r": round(float(raw_row["cumulative_r"]), 4),
        "max_dd_r": round(float(raw_row["max_drawdown_r"]), 4),
        "avg_holding_bars": round(float(raw_row["avg_holding_bars"]), 2),
        "tp1_hit_rate_pct": round(float(raw_row["tp1_hit_rate"]), 2),
        "tp2_hit_rate_pct": round(float(raw_row["tp2_hit_rate"]), 2),
        "long_trades": raw_row["long_trades"],
        "long_r": round(float(raw_row["long_r"]), 4),
        "short_trades": raw_row["short_trades"],
        "short_r": round(float(raw_row["short_r"]), 4),
    }


def build_window_slice_rows(
    *,
    trades: pd.DataFrame,
    start: datetime,
    end: datetime,
    horizon: str,
    labels: list[tuple[str, datetime, datetime]],
    champion: str,
    challenger: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = trades[trades["window"] == PRIMARY_WINDOW].copy()
    for (cost_scenario, strategy_profile), group in subset.groupby(["cost_scenario", "strategy_profile"], sort=False):
        for label, slice_start, slice_end in labels:
            raw_row = summarize_trade_slice(
                group,
                label=label,
                horizon=horizon,
                start=max(slice_start, start),
                end=min(slice_end, end),
            )
            rows.append(
                normalized_slice_row(
                    raw_row=raw_row,
                    cost_scenario=str(cost_scenario),
                    window=PRIMARY_WINDOW,
                    strategy_profile=str(strategy_profile),
                    champion=champion,
                    challenger=challenger,
                )
            )
    return rows


def build_monthly_delta_summary(
    *,
    trades: pd.DataFrame,
    champion: str,
    challenger: str,
) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (cost_scenario, window), window_frame in trades.groupby(["cost_scenario", "window"], sort=False):
        scopes = [
            ("overall", window_frame),
            ("LONG", window_frame[window_frame["side"] == "LONG"]),
            ("SHORT", window_frame[window_frame["side"] == "SHORT"]),
        ]
        for scope, scope_frame in scopes:
            if scope_frame.empty:
                continue
            grouped = (
                scope_frame.groupby(["year", "month", "strategy_profile"])
                .agg(trades=("pnl_r", "size"), cumulative_r=("pnl_r", "sum"))
                .reset_index()
            )
            pnl_pivot = (
                grouped.pivot(index=["year", "month"], columns="strategy_profile", values="cumulative_r")
                .fillna(0.0)
                .reset_index()
            )
            trade_pivot = (
                grouped.pivot(index=["year", "month"], columns="strategy_profile", values="trades")
                .fillna(0)
                .reset_index()
            )
            merged = pnl_pivot.merge(trade_pivot, on=["year", "month"], suffixes=("_r", "_trades"))
            for row in merged.itertuples(index=False):
                champion_r = float(getattr(row, champion + "_r", 0.0))
                challenger_r = float(getattr(row, challenger + "_r", 0.0))
                rows.append(
                    {
                        "cost_scenario": cost_scenario,
                        "window": window,
                        "scope": scope,
                        "year": int(row.year),
                        "month": row.month,
                        "champion_trades": int(getattr(row, champion + "_trades", 0)),
                        "challenger_trades": int(getattr(row, challenger + "_trades", 0)),
                        "champion_cumulative_r": round(champion_r, 4),
                        "challenger_cumulative_r": round(challenger_r, 4),
                        "delta_r": round(challenger_r - champion_r, 4),
                    }
                )
    return rows


def build_offset_summary(monthly_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not monthly_rows:
        return []
    frame = pd.DataFrame(monthly_rows)
    rows: list[dict[str, Any]] = []
    for (cost_scenario, window, scope), group in frame.groupby(["cost_scenario", "window", "scope"], sort=False):
        champion_negative_challenger_positive = group[
            (group["champion_cumulative_r"] < 0) & (group["challenger_cumulative_r"] > 0)
        ]
        challenger_negative_champion_positive = group[
            (group["challenger_cumulative_r"] < 0) & (group["champion_cumulative_r"] > 0)
        ]
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "scope": scope,
                "champion_negative_challenger_positive_months": int(len(champion_negative_challenger_positive)),
                "champion_negative_challenger_positive_delta_r": round(
                    float(champion_negative_challenger_positive["delta_r"].sum()),
                    4,
                ),
                "challenger_negative_champion_positive_months": int(len(challenger_negative_champion_positive)),
                "challenger_negative_champion_positive_delta_r": round(
                    float(challenger_negative_champion_positive["delta_r"].sum()),
                    4,
                ),
            }
        )
    return rows


def build_cost_sensitivity(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary_rows)
    rows: list[dict[str, Any]] = []
    for (_, window, strategy_profile), group in frame.groupby(["overlay_profile", "window", "strategy_profile"], sort=False):
        base = group[group["cost_scenario"] == "base"]
        stress = group[group["cost_scenario"] == "stress_x2"]
        if base.empty or stress.empty:
            continue
        base_row = base.iloc[0]
        stress_row = stress.iloc[0]
        rows.append(
            {
                "window": window,
                "strategy_profile": strategy_profile,
                "candidate_role": base_row["candidate_role"],
                "candidate_label": base_row["candidate_label"],
                "base_trades": int(base_row["trades"]),
                "stress_trades": int(stress_row["trades"]),
                "base_cum_r": round(float(base_row["cum_r"]), 4),
                "stress_cum_r": round(float(stress_row["cum_r"]), 4),
                "delta_cum_r": round(float(stress_row["cum_r"] - base_row["cum_r"]), 4),
                "base_profit_factor": round(float(base_row["profit_factor"]), 4),
                "stress_profit_factor": round(float(stress_row["profit_factor"]), 4),
                "delta_profit_factor": round(float(stress_row["profit_factor"] - base_row["profit_factor"]), 4),
                "base_max_dd_r": round(float(base_row["max_dd_r"]), 4),
                "stress_max_dd_r": round(float(stress_row["max_dd_r"]), 4),
                "delta_max_dd_r": round(float(stress_row["max_dd_r"] - base_row["max_dd_r"]), 4),
            }
        )
    return rows


def build_concentration_summary(
    *,
    rows: list[dict[str, Any]],
    champion: str,
    challenger: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    for horizon, group in frame.groupby("horizon", sort=False):
        base = group[group["cost_scenario"] == "base"].copy()
        if base.empty:
            continue
        pivot = (
            base.pivot(index="label", columns="strategy_profile", values="cum_r")
            .fillna(0.0)
            .reset_index()
        )
        positive_deltas: list[tuple[str, float]] = []
        for row in pivot.itertuples(index=False):
            champion_r = float(getattr(row, champion, 0.0))
            challenger_r = float(getattr(row, challenger, 0.0))
            delta = challenger_r - champion_r
            if delta > 0:
                positive_deltas.append((str(row.label), delta))
        positive_deltas.sort(key=lambda item: item[1], reverse=True)
        total_positive = float(sum(delta for _, delta in positive_deltas))
        top3_share = (sum(delta for _, delta in positive_deltas[:3]) / total_positive * 100.0) if total_positive > 0 else 0.0
        summary.append(
            {
                "horizon": horizon,
                "positive_windows": int(len(positive_deltas)),
                "total_positive_delta_r": round(total_positive, 4),
                "top_positive_label": positive_deltas[0][0] if positive_deltas else None,
                "top_positive_delta_r": round(positive_deltas[0][1], 4) if positive_deltas else None,
                "top3_positive_share_pct": round(top3_share, 2),
                "regime_specialist_tendency": total_positive > 0 and top3_share >= 70.0,
            }
        )
    return summary


def build_promotion_decision(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    champion: str,
    challenger: str,
    overlay_profile: str,
) -> dict[str, Any]:
    summary_map = {
        (row["cost_scenario"], row["window"], row["strategy_profile"]): row
        for row in summary_rows
    }
    side_map = {
        (row["cost_scenario"], row["window"], row["strategy_profile"], row["side"]): row
        for row in side_rows
    }
    concentration_map = {row["horizon"]: row for row in concentration_rows}

    champion_base_primary = summary_map[("base", PRIMARY_WINDOW, champion)]
    challenger_base_primary = summary_map[("base", PRIMARY_WINDOW, challenger)]
    champion_base_secondary_long = side_map[("base", SECONDARY_WINDOW, champion, "LONG")]
    challenger_base_secondary_long = side_map[("base", SECONDARY_WINDOW, challenger, "LONG")]
    champion_base_primary_long = side_map[("base", PRIMARY_WINDOW, champion, "LONG")]
    challenger_base_primary_long = side_map[("base", PRIMARY_WINDOW, challenger, "LONG")]
    champion_base_primary_short = side_map[("base", PRIMARY_WINDOW, champion, "SHORT")]
    challenger_base_primary_short = side_map[("base", PRIMARY_WINDOW, challenger, "SHORT")]
    champion_stress_primary = summary_map[("stress_x2", PRIMARY_WINDOW, champion)]
    challenger_stress_primary = summary_map[("stress_x2", PRIMARY_WINDOW, challenger)]

    pass_base_cum = float(challenger_base_primary["cum_r"]) > float(champion_base_primary["cum_r"])
    pass_base_pf = float(challenger_base_primary["profit_factor"]) > float(champion_base_primary["profit_factor"])
    pass_base_dd = float(challenger_base_primary["max_dd_r"]) <= float(champion_base_primary["max_dd_r"]) + 2.0
    pass_base_long = (
        float(challenger_base_secondary_long["cum_r"]) >= float(champion_base_secondary_long["cum_r"]) - 2.0
    )
    pass_stress_cum = float(challenger_stress_primary["cum_r"]) > float(champion_stress_primary["cum_r"])
    pass_stress_pf = float(challenger_stress_primary["profit_factor"]) >= float(champion_stress_primary["profit_factor"])

    promoted = pass_base_cum and pass_base_pf and pass_base_dd and pass_base_long and pass_stress_cum and pass_stress_pf
    quarter_concentration = concentration_map.get("quarter", {})
    rolling_180_concentration = concentration_map.get("rolling_180", {})
    rolling_365_concentration = concentration_map.get("rolling_365", {})
    regime_specialist = any(
        bool(row.get("regime_specialist_tendency"))
        for row in (quarter_concentration, rolling_180_concentration, rolling_365_concentration)
    )
    promoted_profile = challenger if promoted else champion

    return {
        "champion_profile": champion,
        "challenger_profile": challenger,
        "overlay_profile": overlay_profile,
        "base_primary_window": PRIMARY_WINDOW,
        "secondary_window": SECONDARY_WINDOW,
        "base_champion_primary_cum_r": round(float(champion_base_primary["cum_r"]), 4),
        "base_challenger_primary_cum_r": round(float(challenger_base_primary["cum_r"]), 4),
        "base_primary_delta_cum_r": round(float(challenger_base_primary["cum_r"] - champion_base_primary["cum_r"]), 4),
        "base_champion_primary_profit_factor": round(float(champion_base_primary["profit_factor"]), 4),
        "base_challenger_primary_profit_factor": round(float(challenger_base_primary["profit_factor"]), 4),
        "base_primary_delta_profit_factor": round(
            float(challenger_base_primary["profit_factor"] - champion_base_primary["profit_factor"]),
            4,
        ),
        "base_champion_primary_max_dd_r": round(float(champion_base_primary["max_dd_r"]), 4),
        "base_challenger_primary_max_dd_r": round(float(challenger_base_primary["max_dd_r"]), 4),
        "base_primary_delta_max_dd_r": round(
            float(challenger_base_primary["max_dd_r"] - champion_base_primary["max_dd_r"]),
            4,
        ),
        "base_champion_secondary_long_cum_r": round(float(champion_base_secondary_long["cum_r"]), 4),
        "base_challenger_secondary_long_cum_r": round(float(challenger_base_secondary_long["cum_r"]), 4),
        "base_secondary_long_delta_r": round(
            float(challenger_base_secondary_long["cum_r"] - champion_base_secondary_long["cum_r"]),
            4,
        ),
        "base_primary_long_delta_r": round(
            float(challenger_base_primary_long["cum_r"] - champion_base_primary_long["cum_r"]),
            4,
        ),
        "base_primary_short_delta_r": round(
            float(challenger_base_primary_short["cum_r"] - champion_base_primary_short["cum_r"]),
            4,
        ),
        "stress_champion_primary_cum_r": round(float(champion_stress_primary["cum_r"]), 4),
        "stress_challenger_primary_cum_r": round(float(challenger_stress_primary["cum_r"]), 4),
        "stress_primary_delta_cum_r": round(
            float(challenger_stress_primary["cum_r"] - champion_stress_primary["cum_r"]),
            4,
        ),
        "stress_champion_primary_profit_factor": round(float(champion_stress_primary["profit_factor"]), 4),
        "stress_challenger_primary_profit_factor": round(float(challenger_stress_primary["profit_factor"]), 4),
        "stress_primary_delta_profit_factor": round(
            float(challenger_stress_primary["profit_factor"] - champion_stress_primary["profit_factor"]),
            4,
        ),
        "pass_base_primary_cum_r": pass_base_cum,
        "pass_base_primary_pf": pass_base_pf,
        "pass_base_primary_max_dd": pass_base_dd,
        "pass_base_secondary_long_guard": pass_base_long,
        "pass_stress_primary_cum_r": pass_stress_cum,
        "pass_stress_primary_pf": pass_stress_pf,
        "quarter_top3_positive_share_pct": round(float(quarter_concentration.get("top3_positive_share_pct", 0.0)), 2),
        "rolling_180_top3_positive_share_pct": round(
            float(rolling_180_concentration.get("top3_positive_share_pct", 0.0)),
            2,
        ),
        "rolling_365_top3_positive_share_pct": round(
            float(rolling_365_concentration.get("top3_positive_share_pct", 0.0)),
            2,
        ),
        "regime_specialist_tendency_persists": regime_specialist,
        "promoted_baseline_profile": promoted_profile,
        "promoted_candidate_label": candidate_label(promoted_profile, champion=champion, challenger=challenger),
        "status": "challenger_managed_promoted" if promoted else "champion_managed_retained",
        "next_route": "range-failure",
    }


def build_ordered_matches(champion_df: pd.DataFrame, challenger_df: pd.DataFrame) -> list[MatchPair]:
    champion_df = champion_df.sort_values("signal_time").reset_index(drop=False)
    challenger_df = challenger_df.sort_values("signal_time").reset_index(drop=False)
    champion_times = champion_df["signal_time"].tolist()
    challenger_times = challenger_df["signal_time"].tolist()
    tolerance_seconds = int(MATCH_TOLERANCE.total_seconds())

    def better_score(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        if left[0] != right[0]:
            return left if left[0] > right[0] else right
        return left if left[1] >= right[1] else right

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[int, int]:
        if i >= len(champion_times) or j >= len(challenger_times):
            return (0, 0)

        best = better_score(solve(i + 1, j), solve(i, j + 1))
        gap_seconds = abs(int((champion_times[i] - challenger_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            matched = solve(i + 1, j + 1)
            candidate = (matched[0] + 1, matched[1] - gap_seconds)
            best = better_score(best, candidate)
        return best

    def reconstruct(i: int, j: int) -> list[MatchPair]:
        if i >= len(champion_times) or j >= len(challenger_times):
            return []

        current = solve(i, j)
        gap_seconds = abs(int((champion_times[i] - challenger_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            matched = solve(i + 1, j + 1)
            candidate = (matched[0] + 1, matched[1] - gap_seconds)
            if candidate == current:
                return [
                    MatchPair(
                        champion_idx=int(champion_df.loc[i, "index"]),
                        challenger_idx=int(challenger_df.loc[j, "index"]),
                        kind="exact" if gap_seconds == 0 else "near",
                        gap_hours=gap_seconds / 3600,
                    )
                ] + reconstruct(i + 1, j + 1)

        if solve(i + 1, j) == current:
            return reconstruct(i + 1, j)
        return reconstruct(i, j + 1)

    return reconstruct(0, 0)


def run_pair_diff(
    *,
    output_dir: Path,
    trades: pd.DataFrame,
    champion: str,
    challenger: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = trades[(trades["cost_scenario"] == "base") & (trades["window"] == PRIMARY_WINDOW)].copy()

    pair_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for year in DIFF_YEARS:
        for side in ("LONG", "SHORT"):
            year_side = subset[(subset["year"] == year) & (subset["side"] == side)]
            champion_df = year_side[year_side["strategy_profile"] == champion].sort_values("signal_time")
            challenger_df = year_side[year_side["strategy_profile"] == challenger].sort_values("signal_time")
            pairs = build_ordered_matches(champion_df, challenger_df)
            matched_champion = {pair.champion_idx for pair in pairs}
            matched_challenger = {pair.challenger_idx for pair in pairs}

            for pair in pairs:
                champion_row = subset.loc[pair.champion_idx]
                challenger_row = subset.loc[pair.challenger_idx]
                pair_rows.append(
                    {
                        "year": year,
                        "side": side,
                        "match_kind": pair.kind,
                        "gap_hours": round(pair.gap_hours, 2),
                        "champion_signal_time": champion_row["signal_time"].isoformat(),
                        "challenger_signal_time": challenger_row["signal_time"].isoformat(),
                        "champion_pnl_r": round(float(champion_row["pnl_r"]), 4),
                        "challenger_pnl_r": round(float(challenger_row["pnl_r"]), 4),
                        "delta_r": round(float(challenger_row["pnl_r"] - champion_row["pnl_r"]), 4),
                        "champion_exit_reason": champion_row["exit_reason"],
                        "challenger_exit_reason": challenger_row["exit_reason"],
                        "champion_trend_strength": int(champion_row["trend_strength"]),
                        "challenger_trend_strength": int(challenger_row["trend_strength"]),
                    }
                )

            unmatched_champion = subset.loc[sorted(champion_df.index.difference(list(matched_champion)))]
            unmatched_challenger = subset.loc[sorted(challenger_df.index.difference(list(matched_challenger)))]
            for origin, frame in [("champion_only", unmatched_champion), ("challenger_only", unmatched_challenger)]:
                for row in frame.itertuples():
                    unmatched_rows.append(
                        {
                            "year": year,
                            "side": side,
                            "origin": origin,
                            "signal_time": row.signal_time.isoformat(),
                            "entry_time": row.entry_time.isoformat(),
                            "exit_time": row.exit_time.isoformat(),
                            "trend_strength": int(row.trend_strength),
                            "confidence": int(row.confidence),
                            "bars_held": int(row.bars_held),
                            "exit_reason": row.exit_reason,
                            "pnl_r": round(float(row.pnl_r), 4),
                        }
                    )

            pair_subset = [row for row in pair_rows if row["year"] == year and row["side"] == side]
            summary_rows.append(
                {
                    "year": year,
                    "side": side,
                    "champion_trades": int(len(champion_df)),
                    "challenger_trades": int(len(challenger_df)),
                    "exact_pairs": int(sum(row["match_kind"] == "exact" for row in pair_subset)),
                    "near_pairs": int(sum(row["match_kind"] == "near" for row in pair_subset)),
                    "champion_only": int(len(unmatched_champion)),
                    "challenger_only": int(len(unmatched_challenger)),
                    "matched_delta_r": round(float(sum(row["delta_r"] for row in pair_subset)), 4),
                    "champion_only_pnl_r": round(float(unmatched_champion["pnl_r"].sum()), 4),
                    "challenger_only_pnl_r": round(float(unmatched_challenger["pnl_r"].sum()), 4),
                    "overall_delta_r": round(float(challenger_df["pnl_r"].sum() - champion_df["pnl_r"].sum()), 4),
                }
            )

    monthly_rows: list[dict[str, Any]] = []
    full_primary = subset[subset["year"].isin(DIFF_YEARS)].copy()
    if not full_primary.empty:
        grouped = (
            full_primary.groupby(["year", "month", "strategy_profile"])
            .agg(cumulative_r=("pnl_r", "sum"))
            .reset_index()
        )
        pivot = (
            grouped.pivot(index=["year", "month"], columns="strategy_profile", values="cumulative_r")
            .fillna(0.0)
            .reset_index()
        )
        for row in pivot.itertuples(index=False):
            champion_r = float(getattr(row, champion, 0.0))
            challenger_r = float(getattr(row, challenger, 0.0))
            monthly_rows.append(
                {
                    "year": int(row.year),
                    "month": row.month,
                    "champion_pnl_r": round(champion_r, 4),
                    "challenger_pnl_r": round(challenger_r, 4),
                    "diff_r": round(challenger_r - champion_r, 4),
                }
            )

    write_csv(output_dir / "matched_pairs.csv", pair_rows)
    write_csv(output_dir / "unmatched_trades.csv", unmatched_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "monthly_breakdown.csv", monthly_rows)

    top_positive = sorted(monthly_rows, key=lambda item: item["diff_r"], reverse=True)[:5]
    top_negative = sorted(monthly_rows, key=lambda item: item["diff_r"])[:5]
    report = "\n".join(
        [
            "# Managed Candidate Diff",
            "",
            f"- champion managed: `{champion}`",
            f"- challenger managed: `{challenger}`",
            "",
            "## Summary",
            "",
            render_table(
                summary_rows,
                [
                    ("year", "Year"),
                    ("side", "Side"),
                    ("champion_trades", "Champion Trades"),
                    ("challenger_trades", "Challenger Trades"),
                    ("matched_delta_r", "Matched Delta R"),
                    ("champion_only_pnl_r", "Champion Only R"),
                    ("challenger_only_pnl_r", "Challenger Only R"),
                    ("overall_delta_r", "Overall Delta R"),
                ],
            ),
            "",
            "## Best Months",
            "",
            render_table(
                top_positive,
                [("month", "Month"), ("champion_pnl_r", "Champion"), ("challenger_pnl_r", "Challenger"), ("diff_r", "Diff")],
            ),
            "",
            "## Worst Months",
            "",
            render_table(
                top_negative,
                [("month", "Month"), ("champion_pnl_r", "Champion"), ("challenger_pnl_r", "Challenger"), ("diff_r", "Diff")],
            ),
        ]
    )
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    return summary_rows, pair_rows, unmatched_rows, monthly_rows


def build_report(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    champion: str,
    challenger: str,
    overlay_profile: str,
    output_dir: Path,
) -> str:
    frame_summary = pd.DataFrame(summary_rows)
    frame_side = pd.DataFrame(side_rows)

    base_primary_rows = frame_summary[
        (frame_summary["cost_scenario"] == "base") & (frame_summary["window"] == PRIMARY_WINDOW)
    ].sort_values(["cum_r", "profit_factor"], ascending=[False, False]).to_dict("records")
    stress_primary_rows = frame_summary[
        (frame_summary["cost_scenario"] == "stress_x2") & (frame_summary["window"] == PRIMARY_WINDOW)
    ].sort_values(["cum_r", "profit_factor"], ascending=[False, False]).to_dict("records")
    base_secondary_rows = frame_summary[
        (frame_summary["cost_scenario"] == "base") & (frame_summary["window"] == SECONDARY_WINDOW)
    ].sort_values(["cum_r", "profit_factor"], ascending=[False, False]).to_dict("records")
    base_primary_side_rows = frame_side[
        (frame_side["cost_scenario"] == "base") & (frame_side["window"] == PRIMARY_WINDOW)
    ].sort_values(["side", "strategy_profile"]).to_dict("records")

    primary_offset = next(
        (
            row
            for row in monthly_rows
            if row["cost_scenario"] == "base" and row["window"] == PRIMARY_WINDOW and row["scope"] == "overall"
        ),
        None,
    )
    offset_text = (
        f"主窗口里共有 `{primary_offset['champion_negative_challenger_positive_months']}` 个月出现 `champion 亏 / challenger 盈`，累计 offset `{primary_offset['champion_negative_challenger_positive_delta_r']:.4f}R`。"
        if primary_offset is not None
        else "没有生成月度 offset 诊断。"
    )

    if abs(float(decision_row["base_primary_short_delta_r"])) >= abs(float(decision_row["base_primary_long_delta_r"])):
        alpha_source = (
            f"增量仍主要来自 SHORT，base 主窗口 SHORT delta 为 `{decision_row['base_primary_short_delta_r']:.4f}R`。"
        )
    else:
        alpha_source = (
            f"增量主要来自 LONG，base 主窗口 LONG delta 为 `{decision_row['base_primary_long_delta_r']:.4f}R`。"
        )

    long_guard_text = (
        f"次级窗口 LONG guardrail 保住了，delta = `{decision_row['base_secondary_long_delta_r']:.4f}R`。"
        if bool(decision_row["pass_base_secondary_long_guard"])
        else f"次级窗口 LONG guardrail 没保住，delta = `{decision_row['base_secondary_long_delta_r']:.4f}R`。"
    )
    stress_text = (
        "在 `stress_x2` 下 challenger 仍优于 champion。"
        if bool(decision_row["pass_stress_primary_cum_r"]) and bool(decision_row["pass_stress_primary_pf"])
        else "在 `stress_x2` 下 challenger 没能同时保住 cum_r 和 PF 优势。"
    )
    regime_text = (
        "季度 / rolling 诊断仍显示 `regime-specialist tendency persists`。"
        if bool(decision_row["regime_specialist_tendency_persists"])
        else "季度 / rolling 诊断没有再出现明显的 regime-specialist 集中度。"
    )
    decision_text = (
        "promotion gate 通过，后续主线增强研究默认切到 `challenger_managed`。"
        if decision_row["status"] == "challenger_managed_promoted"
        else "promotion gate 没通过，后续主线增强研究仍保留 `champion_managed`。"
    )

    return "\n".join(
        [
            "# Managed Candidate Promotion Gate",
            "",
            f"- champion managed: `{champion}` + `{overlay_profile}`",
            f"- challenger managed: `{challenger}` + `{overlay_profile}`",
            f"- 主窗口：`{PRIMARY_WINDOW_LABEL}`",
            f"- 次级窗口：`{SECONDARY_WINDOW_LABEL}`",
            "- 成本口径：`base` 使用默认 backtest assumptions，`stress_x2` 固定为 `taker_fee_bps=10 / slippage_bps=4`。",
            "",
            "## Base Primary Window",
            "",
            render_table(
                base_primary_rows,
                [
                    ("candidate_label", "Candidate"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                ],
            ),
            "",
            "## Base Secondary Window",
            "",
            render_table(
                base_secondary_rows,
                [
                    ("candidate_label", "Candidate"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                ],
            ),
            "",
            "## Stress Primary Window",
            "",
            render_table(
                stress_primary_rows,
                [
                    ("candidate_label", "Candidate"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                ],
            ),
            "",
            "## Base Primary Side Split",
            "",
            render_table(
                base_primary_side_rows,
                [
                    ("candidate_label", "Candidate"),
                    ("side", "Side"),
                    ("trades", "Trades"),
                    ("cum_r", "Cum R"),
                    ("expectancy_r", "Exp R"),
                    ("profit_factor", "PF"),
                ],
            ),
            "",
            "## Cost Sensitivity",
            "",
            render_table(
                cost_rows,
                [
                    ("window", "Window"),
                    ("candidate_label", "Candidate"),
                    ("base_cum_r", "Base Cum R"),
                    ("stress_cum_r", "Stress Cum R"),
                    ("delta_cum_r", "Stress-Base Cum R"),
                    ("base_profit_factor", "Base PF"),
                    ("stress_profit_factor", "Stress PF"),
                    ("delta_max_dd_r", "Stress-Base MaxDD"),
                ],
            ),
            "",
            "## Concentration Diagnostics",
            "",
            render_table(
                concentration_rows,
                [
                    ("horizon", "Horizon"),
                    ("positive_windows", "Positive Windows"),
                    ("total_positive_delta_r", "Total Positive Delta R"),
                    ("top_positive_label", "Top Label"),
                    ("top3_positive_share_pct", "Top3 Share %"),
                    ("regime_specialist_tendency", "Regime Specialist"),
                ],
            ),
            "",
            "## Promotion Decision",
            "",
            render_table(
                [decision_row],
                [
                    ("base_primary_delta_cum_r", "Base Delta Cum R"),
                    ("base_primary_delta_profit_factor", "Base Delta PF"),
                    ("base_primary_delta_max_dd_r", "Base Delta MaxDD"),
                    ("base_secondary_long_delta_r", "Base Secondary LONG Delta"),
                    ("stress_primary_delta_cum_r", "Stress Delta Cum R"),
                    ("stress_primary_delta_profit_factor", "Stress Delta PF"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Conclusion",
            "",
            f"- {decision_text}",
            f"- {alpha_source}",
            f"- {long_guard_text}",
            f"- {stress_text}",
            f"- {regime_text}",
            f"- {offset_text}",
            f"- 下一条固定转去 `range-failure`，不是继续扩主线内部 tweak。",
            f"- trade diff: `{output_dir / 'winner_diff' / 'report.md'}`",
        ]
    )


def run_managed_window(
    *,
    service,
    enriched_history: dict[str, dict[str, pd.DataFrame]],
    overlay_profile: str,
    profiles: tuple[str, str],
    symbol: str,
    window_name: str,
    window_label: str,
    start: datetime,
    end: datetime,
    cost_scenario: str,
    champion: str,
    challenger: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    spec = PROFILE_SPEC_MAP[overlay_profile]

    for profile in profiles:
        strategy = service.strategy_service.build_strategy(profile)
        trigger_tf = str(strategy.window_config["trigger_timeframe"])
        if trigger_tf != "1h":
            raise ValueError(f"Managed replay assumes a 1H trigger timeframe, got {trigger_tf} for {profile}.")
        enriched = enriched_history[profile]
        extension_features = precompute_extension_features(trigger_frame=enriched[trigger_tf])
        summary, trades, _ = run_profile(
            service=service,
            strategy=strategy,
            symbol=symbol,
            strategy_profile=profile,
            spec=spec,
            start=start,
            end=end,
            enriched=enriched,
            extension_features=extension_features,
        )
        summary_rows.append(
            summary_row(
                cost_scenario=cost_scenario,
                window=window_name,
                window_label=window_label,
                strategy_profile=profile,
                champion=champion,
                challenger=challenger,
                overlay_profile=overlay_profile,
                summary=summary,
            )
        )
        trade_rows.extend(
            trade_rows_from_trades(
                cost_scenario=cost_scenario,
                window=window_name,
                window_label=window_label,
                strategy_profile=profile,
                champion=champion,
                challenger=challenger,
                overlay_profile=overlay_profile,
                trades=trades,
            )
        )

    return summary_rows, trade_rows


def main() -> None:
    args = parse_args()
    configure_logging()

    champion = args.champion_profile
    challenger = args.challenger_profile
    if champion == challenger:
        raise ValueError("champion-profile and challenger-profile must be different.")
    if args.overlay_profile not in PROFILE_SPEC_MAP:
        raise ValueError(f"Unknown overlay profile: {args.overlay_profile}")

    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_service = build_service()
    profiles = (champion, challenger)
    enriched_history: dict[str, dict[str, pd.DataFrame]] = {}
    for profile in profiles:
        print(f"[promotion-gate] preload {profile}", flush=True)
        enriched_history[profile] = history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=profile,
            start=primary_start,
            end=requested_end,
        )

    resolved_end = resolve_end_from_history(enriched_history)

    all_summary_rows: list[dict[str, Any]] = []
    all_trade_rows: list[dict[str, Any]] = []
    for cost_scenario, overrides in COST_SCENARIOS.items():
        print(f"[promotion-gate] scenario {cost_scenario}", flush=True)
        service = build_service(assumption_overrides=overrides)
        for window_name, window_label, start in (
            (PRIMARY_WINDOW, PRIMARY_WINDOW_LABEL, primary_start),
            (SECONDARY_WINDOW, SECONDARY_WINDOW_LABEL, secondary_start),
        ):
            print(
                f"[promotion-gate] {cost_scenario} {window_name} {start.date().isoformat()}->{resolved_end.date().isoformat()}",
                flush=True,
            )
            summary_rows, trade_rows = run_managed_window(
                service=service,
                enriched_history=enriched_history,
                overlay_profile=args.overlay_profile,
                profiles=profiles,
                symbol=args.symbol,
                window_name=window_name,
                window_label=window_label,
                start=start,
                end=resolved_end,
                cost_scenario=cost_scenario,
                champion=champion,
                challenger=challenger,
            )
            all_summary_rows.extend(summary_rows)
            all_trade_rows.extend(trade_rows)

    trade_frame = build_trade_frame(all_trade_rows)
    side_rows = summarize_trade_distribution(
        trade_frame,
        group_cols=["cost_scenario", "window", "strategy_profile", "candidate_role", "candidate_label", "side"],
    )

    year_rows = build_window_slice_rows(
        trades=trade_frame,
        start=primary_start,
        end=resolved_end,
        horizon="year",
        labels=iter_calendar_windows(primary_start, resolved_end, "year"),
        champion=champion,
        challenger=challenger,
    )
    quarter_rows = build_window_slice_rows(
        trades=trade_frame,
        start=primary_start,
        end=resolved_end,
        horizon="quarter",
        labels=iter_calendar_windows(primary_start, resolved_end, "quarter"),
        champion=champion,
        challenger=challenger,
    )
    rolling_180_rows = build_window_slice_rows(
        trades=trade_frame,
        start=primary_start,
        end=resolved_end,
        horizon="rolling_180",
        labels=iter_rolling_windows(
            primary_start,
            resolved_end,
            window_days=180,
            step_days=args.rolling_180_step_days,
        ),
        champion=champion,
        challenger=challenger,
    )
    rolling_365_rows = build_window_slice_rows(
        trades=trade_frame,
        start=primary_start,
        end=resolved_end,
        horizon="rolling_365",
        labels=iter_rolling_windows(
            primary_start,
            resolved_end,
            window_days=365,
            step_days=args.rolling_365_step_days,
        ),
        champion=champion,
        challenger=challenger,
    )

    monthly_rows = build_monthly_delta_summary(trades=trade_frame, champion=champion, challenger=challenger)
    offset_rows = build_offset_summary(monthly_rows)
    cost_rows = build_cost_sensitivity(all_summary_rows)
    concentration_rows = build_concentration_summary(
        rows=[*quarter_rows, *rolling_180_rows, *rolling_365_rows],
        champion=champion,
        challenger=challenger,
    )
    decision_row = build_promotion_decision(
        summary_rows=all_summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        champion=champion,
        challenger=challenger,
        overlay_profile=args.overlay_profile,
    )

    run_pair_diff(
        output_dir=output_dir / "winner_diff",
        trades=trade_frame,
        champion=champion,
        challenger=challenger,
    )

    report = build_report(
        summary_rows=all_summary_rows,
        side_rows=side_rows,
        cost_rows=cost_rows,
        monthly_rows=offset_rows,
        concentration_rows=concentration_rows,
        decision_row=decision_row,
        champion=champion,
        challenger=challenger,
        overlay_profile=args.overlay_profile,
        output_dir=output_dir,
    )
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")

    write_csv(output_dir / "full_summary.csv", all_summary_rows)
    write_csv(output_dir / "full_side_summary.csv", side_rows)
    write_csv(output_dir / "year_summary.csv", year_rows)
    write_csv(output_dir / "quarter_summary.csv", quarter_rows)
    write_csv(output_dir / "rolling_180_summary.csv", rolling_180_rows)
    write_csv(output_dir / "rolling_365_summary.csv", rolling_365_rows)
    write_csv(output_dir / "monthly_delta_summary.csv", monthly_rows)
    write_csv(output_dir / "offset_summary.csv", offset_rows)
    write_csv(output_dir / "cost_sensitivity.csv", cost_rows)
    write_csv(output_dir / "concentration_summary.csv", concentration_rows)
    write_csv(output_dir / "promotion_decision.csv", [decision_row])
    trade_frame.to_csv(output_dir / "trades_all.csv", index=False)

    print(f"saved promotion gate artifacts to {output_dir}")


if __name__ == "__main__":
    main()
