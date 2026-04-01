from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts/backtests/level_aware_confirmation_compare_full_2020"
OUTPUT_DIR = ROOT / "artifacts/backtests/level_aware_confirmation_bear_diff_2022_2023"

TRADES_CSV = SOURCE_DIR / "backtest_long_scaled1_3_short_fixed1_5_20260323T135105Z_trades.csv"

BASE_PROFILE = "swing_trend_long_regime_gate_v1"
LEVEL_PROFILE = "swing_trend_level_aware_confirmation_v1"
YEARS = (2022, 2023)
MATCH_TOLERANCE = pd.Timedelta(hours=12)


@dataclass(frozen=True)
class MatchPair:
    base_idx: int
    level_idx: int
    kind: str
    gap_hours: float


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trades() -> pd.DataFrame:
    trades = pd.read_csv(
        TRADES_CSV,
        parse_dates=["signal_time", "entry_time", "exit_time"],
    )
    trades = trades[trades["strategy_profile"].isin([BASE_PROFILE, LEVEL_PROFILE])].copy()
    trades = trades[trades["signal_time"].dt.year.isin(YEARS)].copy()
    trades["year"] = trades["signal_time"].dt.year
    trades["month"] = trades["signal_time"].dt.strftime("%Y-%m")
    return trades


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def better_score(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    if left[0] != right[0]:
        return left if left[0] > right[0] else right
    return left if left[1] >= right[1] else right


def build_matches(base_df: pd.DataFrame, level_df: pd.DataFrame) -> list[MatchPair]:
    base_df = base_df.sort_values("signal_time").reset_index(drop=False)
    level_df = level_df.sort_values("signal_time").reset_index(drop=False)
    base_times = base_df["signal_time"].tolist()
    level_times = level_df["signal_time"].tolist()
    tolerance_seconds = int(MATCH_TOLERANCE.total_seconds())

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[int, int]:
        if i >= len(base_times) or j >= len(level_times):
            return (0, 0)

        best = better_score(solve(i + 1, j), solve(i, j + 1))
        gap_seconds = abs(int((base_times[i] - level_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            candidate = solve(i + 1, j + 1)
            candidate = (candidate[0] + 1, candidate[1] - gap_seconds)
            best = better_score(best, candidate)
        return best

    def reconstruct(i: int, j: int) -> list[MatchPair]:
        if i >= len(base_times) or j >= len(level_times):
            return []

        current = solve(i, j)
        gap_seconds = abs(int((base_times[i] - level_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            candidate = solve(i + 1, j + 1)
            candidate = (candidate[0] + 1, candidate[1] - gap_seconds)
            if candidate == current:
                return [
                    MatchPair(
                        base_idx=int(base_df.loc[i, "index"]),
                        level_idx=int(level_df.loc[j, "index"]),
                        kind="exact" if gap_seconds == 0 else "near",
                        gap_hours=gap_seconds / 3600,
                    )
                ] + reconstruct(i + 1, j + 1)

        if solve(i + 1, j) == current:
            return reconstruct(i + 1, j)
        return reconstruct(i, j + 1)

    return reconstruct(0, 0)


def find_blocker(row: pd.Series, other_frame: pd.DataFrame) -> pd.Series | None:
    overlaps = other_frame[
        (other_frame["entry_time"] <= row["entry_time"]) & (other_frame["exit_time"] >= row["entry_time"])
    ]
    if overlaps.empty:
        return None
    return overlaps.sort_values("entry_time").iloc[0]


def summarize_bucket(
    trades: pd.DataFrame,
    *,
    year: int,
    side: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    subset = trades[(trades["year"] == year) & (trades["side"] == side)].copy()
    base_df = subset[subset["strategy_profile"] == BASE_PROFILE].copy()
    level_df = subset[subset["strategy_profile"] == LEVEL_PROFILE].copy()
    pairs = build_matches(base_df, level_df)

    matched_base = {pair.base_idx for pair in pairs}
    matched_level = {pair.level_idx for pair in pairs}

    matched_rows: list[dict[str, Any]] = []
    for pair in pairs:
        base_row = trades.loc[pair.base_idx]
        level_row = trades.loc[pair.level_idx]
        matched_rows.append(
            {
                "year": year,
                "side": side,
                "match_kind": pair.kind,
                "gap_hours": round(pair.gap_hours, 2),
                "base_signal_time": base_row["signal_time"].isoformat(),
                "level_signal_time": level_row["signal_time"].isoformat(),
                "base_entry_time": base_row["entry_time"].isoformat(),
                "level_entry_time": level_row["entry_time"].isoformat(),
                "base_exit_time": base_row["exit_time"].isoformat(),
                "level_exit_time": level_row["exit_time"].isoformat(),
                "base_trend_strength": int(base_row["trend_strength"]),
                "level_trend_strength": int(level_row["trend_strength"]),
                "base_confidence": int(base_row["confidence"]),
                "level_confidence": int(level_row["confidence"]),
                "base_bars_held": int(base_row["bars_held"]),
                "level_bars_held": int(level_row["bars_held"]),
                "base_exit_reason": base_row["exit_reason"],
                "level_exit_reason": level_row["exit_reason"],
                "base_pnl_r": round(float(base_row["pnl_r"]), 4),
                "level_pnl_r": round(float(level_row["pnl_r"]), 4),
                "delta_r": round(float(level_row["pnl_r"] - base_row["pnl_r"]), 4),
            }
        )

    unmatched_rows: list[dict[str, Any]] = []
    base_only = base_df.loc[sorted(set(base_df.index) - matched_base)].copy()
    level_only = level_df.loc[sorted(set(level_df.index) - matched_level)].copy()
    for origin, frame, other_frame in (
        ("base_only", base_only, level_df),
        ("level_only", level_only, base_df),
    ):
        for row in frame.itertuples():
            blocker = find_blocker(pd.Series(row._asdict()), other_frame)
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
                    "blocked_by_open_position": blocker is not None,
                    "blocked_by_signal_time": blocker["signal_time"].isoformat() if blocker is not None else "",
                    "blocked_by_entry_time": blocker["entry_time"].isoformat() if blocker is not None else "",
                    "blocked_by_exit_time": blocker["exit_time"].isoformat() if blocker is not None else "",
                    "blocked_by_exit_reason": str(blocker["exit_reason"]) if blocker is not None else "",
                    "blocked_by_pnl_r": round(float(blocker["pnl_r"]), 4) if blocker is not None else 0.0,
                }
            )

    summary = {
        "year": year,
        "side": side,
        "base_trades": int(len(base_df)),
        "level_trades": int(len(level_df)),
        "shared_pairs": int(len(pairs)),
        "base_only_count": int(len(base_only)),
        "level_only_count": int(len(level_only)),
        "matched_delta_r": round(float(sum(row["delta_r"] for row in matched_rows)), 4),
        "base_only_pnl_r": round(float(base_only["pnl_r"].sum()), 4),
        "level_only_pnl_r": round(float(level_only["pnl_r"].sum()), 4),
        "overall_delta_r": round(float(level_df["pnl_r"].sum() - base_df["pnl_r"].sum()), 4),
    }
    return matched_rows, unmatched_rows, summary


def build_monthly_rows(trades: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        trades.groupby(["month", "strategy_profile"])["pnl_r"]
        .agg(["sum", "count"])
        .reset_index()
    )
    pivot_sum = grouped.pivot(index="month", columns="strategy_profile", values="sum").fillna(0.0)
    pivot_count = grouped.pivot(index="month", columns="strategy_profile", values="count").fillna(0)
    rows: list[dict[str, Any]] = []
    for month in pivot_sum.index:
        base_r = float(pivot_sum.loc[month].get(BASE_PROFILE, 0.0))
        level_r = float(pivot_sum.loc[month].get(LEVEL_PROFILE, 0.0))
        rows.append(
            {
                "month": month,
                "base_trades": int(pivot_count.loc[month].get(BASE_PROFILE, 0)),
                "level_trades": int(pivot_count.loc[month].get(LEVEL_PROFILE, 0)),
                "base_pnl_r": round(base_r, 4),
                "level_pnl_r": round(level_r, 4),
                "delta_r": round(level_r - base_r, 4),
            }
        )
    return rows


def top_rows(rows: list[dict[str, Any]], *, key: str, reverse: bool, limit: int = 8) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: item[key], reverse=reverse)[:limit]


def build_report(
    *,
    summaries: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Level-Aware Confirmation 2022-2023 Diff",
        "",
        "这份 diff 只看 `2022-2023`，目的是回答一件更窄的事：`level-aware confirmation` 为什么在熊市阶段比当前主线更差。",
        "",
        "最重要的结论不是“共享交易整体变差”，而是：",
        "",
        "- `2022` 的拖累主要发生在 `SHORT`，共享交易其实略有改善，但 `level-aware` 跳过了一批更早、更好的 short，又放出了一批更晚、更差的 short。",
        "- `2023` 的拖累主要发生在 `LONG`，共享交易只小幅变差，真正的损失来自它错过了几笔原主线的正收益 long，同时新增了几笔后续 stop-loss long。",
        "- 所以它更像一种 `trade-set rewrite`，不是单纯提高确认质量。",
        "",
        "## 分桶摘要",
        "",
        "| Year | Side | Base Trades | Level Trades | Shared | Base Only | Level Only | Matched Delta R | Base Only R | Level Only R | Overall Delta R |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['year']} | {row['side']} | {row['base_trades']} | {row['level_trades']} | "
            f"{row['shared_pairs']} | {row['base_only_count']} | {row['level_only_count']} | "
            f"{row['matched_delta_r']:.4f} | {row['base_only_pnl_r']:.4f} | {row['level_only_pnl_r']:.4f} | "
            f"{row['overall_delta_r']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `2022 SHORT`：共享部分 `+2.3134R`，说明配对上的 short 没更差；真正的问题是 `base_only +0.3295R` 而 `level_only -2.5749R`。也就是 level-aware 没把空头确认做得更稳，反而把一部分更好的早期 short 换成了更差的后续 short。",
            "- `2023 LONG`：共享部分只 `-0.2848R`，真正拖累来自 `base_only +1.7281R` 和 `level_only -1.8407R`。这不是确认质量全面退化，而是它跳过了几笔原主线的正收益 long，又新放了几笔后续 long stop-loss。",
            "",
            "## 最差月份",
            "",
            markdown_table(
                top_rows(monthly_rows, key='delta_r', reverse=False, limit=8),
                [
                    ('month', 'Month'),
                    ('base_trades', 'Base Trades'),
                    ('level_trades', 'Level Trades'),
                    ('base_pnl_r', 'Base R'),
                    ('level_pnl_r', 'Level R'),
                    ('delta_r', 'Delta R'),
                ],
            ),
            "",
            "## 最关键的配对变化",
            "",
            "正向贡献最大：",
            "",
            markdown_table(
                top_rows(matched_rows, key='delta_r', reverse=True, limit=6),
                [
                    ('year', 'Year'),
                    ('side', 'Side'),
                    ('gap_hours', 'Gap H'),
                    ('base_signal_time', 'Base Signal'),
                    ('level_signal_time', 'Level Signal'),
                    ('base_exit_reason', 'Base Exit'),
                    ('level_exit_reason', 'Level Exit'),
                    ('base_pnl_r', 'Base R'),
                    ('level_pnl_r', 'Level R'),
                    ('delta_r', 'Delta R'),
                ],
            ),
            "",
            "负向贡献最大：",
            "",
            markdown_table(
                top_rows(matched_rows, key='delta_r', reverse=False, limit=6),
                [
                    ('year', 'Year'),
                    ('side', 'Side'),
                    ('gap_hours', 'Gap H'),
                    ('base_signal_time', 'Base Signal'),
                    ('level_signal_time', 'Level Signal'),
                    ('base_exit_reason', 'Base Exit'),
                    ('level_exit_reason', 'Level Exit'),
                    ('base_pnl_r', 'Base R'),
                    ('level_pnl_r', 'Level R'),
                    ('delta_r', 'Delta R'),
                ],
            ),
            "",
            "## Base-Only 正收益样本",
            "",
            markdown_table(
                top_rows(
                    [row for row in unmatched_rows if row['origin'] == 'base_only'],
                    key='pnl_r',
                    reverse=True,
                    limit=8,
                ),
                [
                    ('year', 'Year'),
                    ('side', 'Side'),
                    ('signal_time', 'Signal'),
                    ('exit_reason', 'Exit'),
                    ('pnl_r', 'Pnl R'),
                    ('blocked_by_open_position', 'Blocked'),
                    ('blocked_by_signal_time', 'Blocked By Signal'),
                    ('blocked_by_pnl_r', 'Blocked By R'),
                ],
            ),
            "",
            "## Level-Only 负收益样本",
            "",
            markdown_table(
                top_rows(
                    [row for row in unmatched_rows if row['origin'] == 'level_only'],
                    key='pnl_r',
                    reverse=False,
                    limit=8,
                ),
                [
                    ('year', 'Year'),
                    ('side', 'Side'),
                    ('signal_time', 'Signal'),
                    ('exit_reason', 'Exit'),
                    ('pnl_r', 'Pnl R'),
                    ('blocked_by_open_position', 'Blocked'),
                    ('blocked_by_signal_time', 'Blocked By Signal'),
                    ('blocked_by_pnl_r', 'Blocked By R'),
                ],
            ),
            "",
            "## 工程判断",
            "",
            "- 这条分支现在不应被理解成“熊市下确认更严格所以更稳”。当前证据更支持相反说法：它在 `2022-2023` 会把一部分早期有效交易改写成更晚、更差的交易集合。",
            "- 如果后面还要保留它，最合理的定位不是全局主线升级，而是最多作为 `bull-regime` 候选变体去看。",
            "- 在当前证据下，不值得继续调 proximity/min_hits 去救这条线；那大概率只会把 trade-set rewrite 做得更严重。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ensure_output_dir()
    trades = load_trades()

    all_matched_rows: list[dict[str, Any]] = []
    all_unmatched_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for year, side in ((2022, "SHORT"), (2023, "LONG")):
        matched_rows, unmatched_rows, summary = summarize_bucket(trades, year=year, side=side)
        all_matched_rows.extend(matched_rows)
        all_unmatched_rows.extend(unmatched_rows)
        all_summaries.append(summary)

    monthly_rows = build_monthly_rows(trades)

    write_csv(OUTPUT_DIR / "matched_pairs.csv", all_matched_rows)
    write_csv(OUTPUT_DIR / "unmatched_trades.csv", all_unmatched_rows)
    write_csv(OUTPUT_DIR / "bucket_summary.csv", all_summaries)
    write_csv(OUTPUT_DIR / "monthly_breakdown.csv", monthly_rows)

    report = build_report(
        summaries=all_summaries,
        matched_rows=all_matched_rows,
        unmatched_rows=all_unmatched_rows,
        monthly_rows=monthly_rows,
    )
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(f"Saved matched pairs CSV: {OUTPUT_DIR / 'matched_pairs.csv'}")
    print(f"Saved unmatched trades CSV: {OUTPUT_DIR / 'unmatched_trades.csv'}")
    print(f"Saved bucket summary CSV: {OUTPUT_DIR / 'bucket_summary.csv'}")
    print(f"Saved monthly breakdown CSV: {OUTPUT_DIR / 'monthly_breakdown.csv'}")
    print(f"Saved report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
