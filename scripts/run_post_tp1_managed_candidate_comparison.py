from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SOURCE_DIR = ROOT / "artifacts" / "backtests" / "post_tp1_extension_dual_baseline"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "post_tp1_managed_candidate_comparison"
DEFAULT_CHAMPION = "swing_trend_long_regime_gate_v1"
DEFAULT_CHALLENGER = "swing_trend_long_regime_short_no_reversal_no_aux_v1"
PRIMARY_WINDOW = "full_2020"
SECONDARY_WINDOW = "two_year"
DIFF_YEARS = (2022, 2024, 2026)
MATCH_TOLERANCE = pd.Timedelta(hours=12)


@dataclass(frozen=True)
class MatchPair:
    champion_idx: int
    challenger_idx: int
    kind: str
    gap_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare managed champion vs managed challenger using fixed post-TP1 overlay artifacts.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--champion-profile", default=DEFAULT_CHAMPION)
    parser.add_argument("--challenger-profile", default=DEFAULT_CHALLENGER)
    parser.add_argument("--overlay-profile", default=None, help="Defaults to selected_overlay_profile from overlay_classification.csv")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def role_label(profile: str, *, champion: str, challenger: str) -> str:
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
        body.append("| " + " | ".join(str(row.get(key)) for key, _ in columns) + " |")
    return "\n".join([head, sep, *body])


def load_overlay_profile(source_dir: Path, requested_overlay: str | None) -> str:
    if requested_overlay:
        return requested_overlay
    classification_path = source_dir / "overlay_classification.csv"
    classification = pd.read_csv(classification_path)
    if classification.empty:
        raise ValueError(f"overlay_classification.csv is empty: {classification_path}")
    overlay_profile = str(classification.iloc[0]["selected_overlay_profile"]).strip()
    if not overlay_profile or overlay_profile.lower() == "nan":
        raise ValueError(f"No selected overlay profile found in {classification_path}")
    return overlay_profile


def load_trade_frame(source_dir: Path, *, champion: str, challenger: str, overlay_profile: str) -> pd.DataFrame:
    trades = pd.read_csv(source_dir / "trades_all.csv")
    trades = trades[
        trades["baseline_strategy_profile"].isin([champion, challenger]) & trades["profile"].eq(overlay_profile)
    ].copy()
    if trades.empty:
        raise ValueError("No trades found for the requested managed candidates.")
    for column in ("signal_time", "entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    trades["year"] = trades["entry_time"].dt.year.astype(int)
    trades["month"] = trades["entry_time"].dt.strftime("%Y-%m")
    return trades


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
                "cumulative_r": round(float(group["pnl_r"].sum()), 4),
                "profit_factor": profit_factor,
                "max_drawdown_r": compute_group_max_drawdown(group),
                "avg_bars_held": round(float(group["bars_held"].mean()), 2),
            }
        )
        rows.append(row)
    return rows


def build_monthly_delta_summary(
    *,
    trades: pd.DataFrame,
    champion: str,
    challenger: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window, window_frame in trades.groupby("window", sort=False):
        scopes = [
            ("overall", window_frame),
            ("LONG", window_frame[window_frame["side"] == "LONG"]),
            ("SHORT", window_frame[window_frame["side"] == "SHORT"]),
        ]
        for scope, scope_frame in scopes:
            if scope_frame.empty:
                continue
            grouped = (
                scope_frame.groupby(["year", "month", "baseline_strategy_profile"])
                .agg(trades=("pnl_r", "size"), cumulative_r=("pnl_r", "sum"))
                .reset_index()
            )
            pnl_pivot = (
                grouped.pivot(index=["year", "month"], columns="baseline_strategy_profile", values="cumulative_r")
                .fillna(0.0)
                .reset_index()
            )
            trade_pivot = (
                grouped.pivot(index=["year", "month"], columns="baseline_strategy_profile", values="trades")
                .fillna(0)
                .reset_index()
            )
            merged = pnl_pivot.merge(trade_pivot, on=["year", "month"], suffixes=("_r", "_trades"))
            for row in merged.itertuples(index=False):
                champion_r = float(getattr(row, champion + "_r", 0.0))
                challenger_r = float(getattr(row, challenger + "_r", 0.0))
                rows.append(
                    {
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
    for (window, scope), group in frame.groupby(["window", "scope"], sort=False):
        champion_negative_challenger_positive = group[
            (group["champion_cumulative_r"] < 0) & (group["challenger_cumulative_r"] > 0)
        ]
        challenger_negative_champion_positive = group[
            (group["challenger_cumulative_r"] < 0) & (group["champion_cumulative_r"] > 0)
        ]
        rows.append(
            {
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


def build_comparison_decision(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    champion: str,
    challenger: str,
    primary_window: str = PRIMARY_WINDOW,
    secondary_window: str = SECONDARY_WINDOW,
) -> dict[str, Any]:
    by_window_profile = {
        (row["window"], row["baseline_strategy_profile"]): row
        for row in summary_rows
    }
    by_window_side = {
        (row["window"], row["baseline_strategy_profile"], row["side"]): row
        for row in side_rows
    }
    champion_primary = by_window_profile[(primary_window, champion)]
    challenger_primary = by_window_profile[(primary_window, challenger)]
    champion_secondary_long = by_window_side[(secondary_window, champion, "LONG")]
    challenger_secondary_long = by_window_side[(secondary_window, challenger, "LONG")]
    champion_primary_long = by_window_side[(primary_window, champion, "LONG")]
    challenger_primary_long = by_window_side[(primary_window, challenger, "LONG")]
    champion_primary_short = by_window_side[(primary_window, champion, "SHORT")]
    challenger_primary_short = by_window_side[(primary_window, challenger, "SHORT")]

    pass_cum = float(challenger_primary["cum_r"]) > float(champion_primary["cum_r"])
    pass_pf = float(challenger_primary["profit_factor"]) > float(champion_primary["profit_factor"])
    pass_dd = float(challenger_primary["max_dd_r"]) <= float(champion_primary["max_dd_r"]) + 2.0
    pass_long_guard = float(challenger_secondary_long["cum_r"]) >= float(champion_secondary_long["cum_r"]) - 2.0
    preferred = pass_cum and pass_pf and pass_dd and pass_long_guard

    return {
        "champion_profile": champion,
        "challenger_profile": challenger,
        "primary_window": primary_window,
        "secondary_window": secondary_window,
        "champion_primary_cum_r": round(float(champion_primary["cum_r"]), 4),
        "challenger_primary_cum_r": round(float(challenger_primary["cum_r"]), 4),
        "primary_delta_cum_r": round(float(challenger_primary["cum_r"] - champion_primary["cum_r"]), 4),
        "champion_primary_profit_factor": round(float(champion_primary["profit_factor"]), 4),
        "challenger_primary_profit_factor": round(float(challenger_primary["profit_factor"]), 4),
        "primary_delta_profit_factor": round(
            float(challenger_primary["profit_factor"] - champion_primary["profit_factor"]),
            4,
        ),
        "champion_primary_max_dd_r": round(float(champion_primary["max_dd_r"]), 4),
        "challenger_primary_max_dd_r": round(float(challenger_primary["max_dd_r"]), 4),
        "primary_delta_max_dd_r": round(float(challenger_primary["max_dd_r"] - champion_primary["max_dd_r"]), 4),
        "champion_secondary_long_cum_r": round(float(champion_secondary_long["cum_r"]), 4),
        "challenger_secondary_long_cum_r": round(float(challenger_secondary_long["cum_r"]), 4),
        "secondary_long_delta_r": round(
            float(challenger_secondary_long["cum_r"] - champion_secondary_long["cum_r"]),
            4,
        ),
        "primary_long_delta_r": round(
            float(challenger_primary_long["cum_r"] - champion_primary_long["cum_r"]),
            4,
        ),
        "primary_short_delta_r": round(
            float(challenger_primary_short["cum_r"] - champion_primary_short["cum_r"]),
            4,
        ),
        "pass_primary_cum_r": pass_cum,
        "pass_primary_pf": pass_pf,
        "pass_primary_max_dd": pass_dd,
        "pass_secondary_long_guard": pass_long_guard,
        "status": "challenger_managed_preferred" if preferred else "champion_managed_retained",
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

    return reconstruct(0, j=0)


def run_pair_diff(
    *,
    output_dir: Path,
    trades: pd.DataFrame,
    champion: str,
    challenger: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = trades[trades["window"] == PRIMARY_WINDOW].copy()

    pair_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for year in DIFF_YEARS:
        for side in ("LONG", "SHORT"):
            year_side = subset[(subset["year"] == year) & (subset["side"] == side)]
            champion_df = year_side[year_side["baseline_strategy_profile"] == champion].sort_values("signal_time")
            challenger_df = year_side[year_side["baseline_strategy_profile"] == challenger].sort_values("signal_time")
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
            full_primary.groupby(["year", "month", "baseline_strategy_profile"])
            .agg(cumulative_r=("pnl_r", "sum"))
            .reset_index()
        )
        pivot = (
            grouped.pivot(index=["year", "month"], columns="baseline_strategy_profile", values="cumulative_r")
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


def main() -> None:
    args = parse_args()
    champion = args.champion_profile
    challenger = args.challenger_profile
    if champion == challenger:
        raise ValueError("champion-profile and challenger-profile must be different.")

    source_dir = resolve_path(args.source_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_profile = load_overlay_profile(source_dir, args.overlay_profile)
    summary_all = pd.read_csv(source_dir / "summary_all.csv")
    side_all = pd.read_csv(source_dir / "side_summary_all.csv")
    managed_summary = summary_all[
        summary_all["baseline_strategy_profile"].isin([champion, challenger]) & summary_all["profile"].eq(overlay_profile)
    ].copy()
    managed_side = side_all[
        side_all["baseline_strategy_profile"].isin([champion, challenger]) & side_all["profile"].eq(overlay_profile)
    ].copy()
    if managed_summary.empty or managed_side.empty:
        raise ValueError("Managed summary/side rows are empty for the selected overlay.")

    managed_summary["candidate_label"] = managed_summary["baseline_strategy_profile"].map(
        lambda item: role_label(item, champion=champion, challenger=challenger)
    )
    managed_side["candidate_label"] = managed_side["baseline_strategy_profile"].map(
        lambda item: role_label(item, champion=champion, challenger=challenger)
    )

    trades = load_trade_frame(source_dir, champion=champion, challenger=challenger, overlay_profile=overlay_profile)
    year_rows = summarize_trade_distribution(
        trades,
        group_cols=["window", "baseline_strategy_profile", "year"],
    )
    for row in year_rows:
        row["candidate_label"] = role_label(
            str(row["baseline_strategy_profile"]),
            champion=champion,
            challenger=challenger,
        )

    monthly_rows = build_monthly_delta_summary(trades=trades, champion=champion, challenger=challenger)
    offset_rows = build_offset_summary(monthly_rows)
    decision_row = build_comparison_decision(
        summary_rows=managed_summary.to_dict("records"),
        side_rows=managed_side.to_dict("records"),
        champion=champion,
        challenger=challenger,
    )

    diff_summary_rows, _, _, _ = run_pair_diff(
        output_dir=output_dir / "winner_diff",
        trades=trades,
        champion=champion,
        challenger=challenger,
    )

    primary_summary_rows = managed_summary[managed_summary["window"] == PRIMARY_WINDOW].sort_values(
        ["cum_r", "profit_factor"],
        ascending=[False, False],
    )
    secondary_summary_rows = managed_summary[managed_summary["window"] == SECONDARY_WINDOW].sort_values(
        ["cum_r", "profit_factor"],
        ascending=[False, False],
    )
    primary_side_rows = managed_side[managed_side["window"] == PRIMARY_WINDOW].sort_values(
        ["side", "baseline_strategy_profile"],
    )

    short_delta = float(decision_row["primary_short_delta_r"])
    long_delta = float(decision_row["primary_long_delta_r"])
    if abs(short_delta) >= abs(long_delta):
        alpha_source = f"主窗口差值几乎全部来自 SHORT，challenger managed 比 champion managed 多了 `{short_delta:.4f}R`。"
    else:
        alpha_source = f"主窗口差值主要来自 LONG，challenger managed 比 champion managed 多了 `{long_delta:.4f}R`。"

    if abs(long_delta) <= 1e-4:
        long_text = "两条管理层候选在 LONG 侧几乎完全一样，这次对照本质上是在看 overlay 生效后，SHORT 侧 alpha 排名有没有改变。"
    else:
        long_text = f"LONG 侧也出现了差值，主窗口 LONG delta 为 `{long_delta:.4f}R`。"

    status = str(decision_row["status"])
    if status == "challenger_managed_preferred":
        decision_text = "固定口径下应优先保留 `challenger + universal_overlay` 作为更强的管理层候选。"
    else:
        decision_text = "固定口径下 challenger managed 没能稳稳压过 champion managed，默认仍保留 champion managed。"

    primary_offset = next(
        (
            row
            for row in offset_rows
            if row["window"] == PRIMARY_WINDOW and row["scope"] == "overall"
        ),
        None,
    )
    offset_text = (
        f"主窗口里共有 `{primary_offset['champion_negative_challenger_positive_months']}` 个月出现 `champion 亏 / challenger 盈`，累计 offset `{primary_offset['champion_negative_challenger_positive_delta_r']:.4f}R`。"
        if primary_offset is not None
        else "没有生成月度 offset 诊断。"
    )

    report_lines = [
        "# Managed Candidate Fixed Comparison",
        "",
        f"- source dir: `{source_dir}`",
        f"- overlay profile: `{overlay_profile}`",
        f"- champion managed: `{champion}` + `{overlay_profile}`",
        f"- challenger managed: `{challenger}` + `{overlay_profile}`",
        "- 说明：这次不重跑 backtest，直接复用正式 dual-baseline exit alpha 产物做固定口径对照。",
        "",
        "## Primary Window",
        "",
        render_table(
            primary_summary_rows.to_dict("records"),
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
        "## Secondary Window",
        "",
        render_table(
            secondary_summary_rows.to_dict("records"),
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
        "## Primary Side Split",
        "",
        render_table(
            primary_side_rows.to_dict("records"),
            [
                ("candidate_label", "Candidate"),
                ("side", "Side"),
                ("trades", "Trades"),
                ("cum_r", "Cum R"),
                ("avg_r", "Avg R"),
                ("profit_factor", "PF"),
            ],
        ),
        "",
        "## Decision",
        "",
        render_table(
            [decision_row],
            [
                ("primary_delta_cum_r", "Primary Delta Cum R"),
                ("primary_delta_profit_factor", "Primary Delta PF"),
                ("primary_delta_max_dd_r", "Primary Delta MaxDD"),
                ("secondary_long_delta_r", "Secondary LONG Delta"),
                ("primary_short_delta_r", "Primary SHORT Delta"),
                ("status", "Status"),
            ],
        ),
        "",
        "## Conclusion",
        "",
        f"- {decision_text}",
        f"- {alpha_source}",
        f"- {long_text}",
        f"- {offset_text}",
        f"- trade diff: `{output_dir / 'winner_diff' / 'report.md'}`",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    write_csv(output_dir / "managed_candidate_summary.csv", managed_summary.to_dict("records"))
    write_csv(output_dir / "managed_candidate_side_summary.csv", managed_side.to_dict("records"))
    write_csv(output_dir / "managed_candidate_year_summary.csv", year_rows)
    write_csv(output_dir / "monthly_delta_summary.csv", monthly_rows)
    write_csv(output_dir / "offset_summary.csv", offset_rows)
    write_csv(output_dir / "comparison_decision.csv", [decision_row])

    print(f"saved managed candidate comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
