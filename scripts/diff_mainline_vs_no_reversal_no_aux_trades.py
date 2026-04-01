from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SOURCE_DIR = ROOT / "artifacts/backtests/entry_attribution_matrix_2020"
OUTPUT_DIR = ROOT / "artifacts/backtests/mainline_vs_no_reversal_no_aux_trade_diff_2022_2024"

MAINLINE = "entry_attr_r1_rf1_hs1_aux1"
NO_REV_NO_AUX = "entry_attr_r0_rf1_hs1_aux0"
YEARS = (2022, 2024)
MATCH_TOLERANCE = pd.Timedelta(hours=12)

PROFILE_LABELS = {
    MAINLINE: "Mainline (R1 RF1 HS1 AUX1)",
    NO_REV_NO_AUX: "No-Reversal-No-Aux (R0 RF1 HS1 AUX0)",
}


@dataclass(frozen=True)
class MatchPair:
    main_idx: int
    alt_idx: int
    kind: str
    gap_hours: float


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trades() -> pd.DataFrame:
    trades = pd.read_csv(
        SOURCE_DIR / "entry_attribution_oos_trades.csv",
        parse_dates=["signal_time", "entry_time", "exit_time"],
    )
    trades = trades[trades["strategy_profile"].isin([MAINLINE, NO_REV_NO_AUX])].copy()
    trades["year"] = trades["entry_time"].dt.year
    trades["month"] = trades["entry_time"].dt.strftime("%Y-%m")
    trades["profile_label"] = trades["strategy_profile"].map(PROFILE_LABELS)
    return trades


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def better_score(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    if left[0] != right[0]:
        return left if left[0] > right[0] else right
    return left if left[1] >= right[1] else right


def build_ordered_matches(main_df: pd.DataFrame, alt_df: pd.DataFrame) -> list[MatchPair]:
    main_df = main_df.sort_values("signal_time").reset_index(drop=False)
    alt_df = alt_df.sort_values("signal_time").reset_index(drop=False)
    main_times = main_df["signal_time"].tolist()
    alt_times = alt_df["signal_time"].tolist()
    tolerance_seconds = int(MATCH_TOLERANCE.total_seconds())

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[int, int]:
        if i >= len(main_times) or j >= len(alt_times):
            return (0, 0)

        best = better_score(solve(i + 1, j), solve(i, j + 1))
        gap_seconds = abs(int((main_times[i] - alt_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            matched = solve(i + 1, j + 1)
            candidate = (matched[0] + 1, matched[1] - gap_seconds)
            best = better_score(best, candidate)
        return best

    def reconstruct(i: int, j: int) -> list[MatchPair]:
        if i >= len(main_times) or j >= len(alt_times):
            return []

        current = solve(i, j)

        gap_seconds = abs(int((main_times[i] - alt_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            matched = solve(i + 1, j + 1)
            candidate = (matched[0] + 1, matched[1] - gap_seconds)
            if candidate == current:
                gap_hours = gap_seconds / 3600
                return [
                    MatchPair(
                        main_idx=int(main_df.loc[i, "index"]),
                        alt_idx=int(alt_df.loc[j, "index"]),
                        kind="exact" if gap_seconds == 0 else "near",
                        gap_hours=gap_hours,
                    )
                ] + reconstruct(i + 1, j + 1)

        if solve(i + 1, j) == current:
            return reconstruct(i + 1, j)
        return reconstruct(i, j + 1)

    return reconstruct(0, 0)


def summarize_pairs(
    *,
    trades: pd.DataFrame,
    year: int,
    side: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    subset = trades[(trades["year"] == year) & (trades["side"] == side)].copy()
    main_df = subset[subset["strategy_profile"] == MAINLINE].sort_values("signal_time")
    alt_df = subset[subset["strategy_profile"] == NO_REV_NO_AUX].sort_values("signal_time")
    pairs = build_ordered_matches(main_df, alt_df)

    matched_main = {pair.main_idx for pair in pairs}
    matched_alt = {pair.alt_idx for pair in pairs}

    pair_rows: list[dict[str, object]] = []
    for pair in pairs:
        main_row = trades.loc[pair.main_idx]
        alt_row = trades.loc[pair.alt_idx]
        pair_rows.append(
            {
                "year": year,
                "side": side,
                "match_kind": pair.kind,
                "gap_hours": round(pair.gap_hours, 2),
                "main_signal_time": main_row["signal_time"].isoformat(),
                "alt_signal_time": alt_row["signal_time"].isoformat(),
                "main_entry_time": main_row["entry_time"].isoformat(),
                "alt_entry_time": alt_row["entry_time"].isoformat(),
                "main_exit_time": main_row["exit_time"].isoformat(),
                "alt_exit_time": alt_row["exit_time"].isoformat(),
                "main_trend_strength": int(main_row["trend_strength"]),
                "alt_trend_strength": int(alt_row["trend_strength"]),
                "main_confidence": int(main_row["confidence"]),
                "alt_confidence": int(alt_row["confidence"]),
                "main_bars_held": int(main_row["bars_held"]),
                "alt_bars_held": int(alt_row["bars_held"]),
                "main_exit_reason": main_row["exit_reason"],
                "alt_exit_reason": alt_row["exit_reason"],
                "main_pnl_r": round(float(main_row["pnl_r"]), 4),
                "alt_pnl_r": round(float(alt_row["pnl_r"]), 4),
                "delta_r": round(float(alt_row["pnl_r"] - main_row["pnl_r"]), 4),
            }
        )

    unmatched_rows: list[dict[str, object]] = []
    unmatched_main = trades.loc[sorted(main_df.index.difference(list(matched_main)))]
    unmatched_alt = trades.loc[sorted(alt_df.index.difference(list(matched_alt)))]

    for origin, frame in [("mainline_only", unmatched_main), ("no_reversal_no_aux_only", unmatched_alt)]:
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

    summary = {
        "year": year,
        "side": side,
        "main_trades": int(len(main_df)),
        "no_reversal_no_aux_trades": int(len(alt_df)),
        "exact_pairs": int(sum(pair.kind == "exact" for pair in pairs)),
        "near_pairs": int(sum(pair.kind == "near" for pair in pairs)),
        "mainline_only": int(len(unmatched_main)),
        "no_reversal_no_aux_only": int(len(unmatched_alt)),
        "matched_main_pnl_r": round(float(sum(row["main_pnl_r"] for row in pair_rows)), 4),
        "matched_no_reversal_no_aux_pnl_r": round(float(sum(row["alt_pnl_r"] for row in pair_rows)), 4),
        "matched_delta_r": round(float(sum(row["delta_r"] for row in pair_rows)), 4),
        "mainline_only_pnl_r": round(float(unmatched_main["pnl_r"].sum()), 4),
        "no_reversal_no_aux_only_pnl_r": round(float(unmatched_alt["pnl_r"].sum()), 4),
        "overall_delta_r": round(float(alt_df["pnl_r"].sum() - main_df["pnl_r"].sum()), 4),
    }
    return pair_rows, unmatched_rows, summary


def build_monthly_rows(trades: pd.DataFrame) -> list[dict[str, object]]:
    subset = trades[trades["year"].isin(YEARS)].copy()
    grouped = (
        subset.groupby(["year", "month", "strategy_profile"])
        .agg(trades=("pnl_r", "size"), cum_r=("pnl_r", "sum"))
        .reset_index()
    )
    pivot = grouped.pivot(index=["year", "month"], columns="strategy_profile", values="cum_r").fillna(0.0).reset_index()
    rows: list[dict[str, object]] = []
    for row in pivot.itertuples(index=False):
        main_r = float(getattr(row, MAINLINE, 0.0))
        alt_r = float(getattr(row, NO_REV_NO_AUX, 0.0))
        rows.append(
            {
                "year": int(row.year),
                "month": row.month,
                "mainline_pnl_r": round(main_r, 4),
                "no_reversal_no_aux_pnl_r": round(alt_r, 4),
                "diff_r": round(alt_r - main_r, 4),
            }
        )
    return rows


def top_rows(rows: list[dict[str, object]], *, key: str, reverse: bool, limit: int = 5) -> list[dict[str, object]]:
    return sorted(rows, key=lambda item: item[key], reverse=reverse)[:limit]


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def format_summary_row(row: dict[str, object]) -> str:
    return (
        f"| {row['year']} | {row['side']} | {row['main_trades']} | {row['no_reversal_no_aux_trades']} | "
        f"{row['exact_pairs']} | {row['near_pairs']} | {row['mainline_only']} | {row['no_reversal_no_aux_only']} | "
        f"{row['matched_delta_r']:.4f} | {row['mainline_only_pnl_r']:.4f} | {row['no_reversal_no_aux_only_pnl_r']:.4f} | "
        f"{row['overall_delta_r']:.4f} |"
    )


def format_month_examples(rows: list[dict[str, object]]) -> str:
    lines = []
    for year in YEARS:
        year_rows = [row for row in rows if row["year"] == year]
        best_alt = top_rows([row for row in year_rows if row["diff_r"] > 0], key="diff_r", reverse=True, limit=3)
        best_main = top_rows([row for row in year_rows if row["diff_r"] < 0], key="diff_r", reverse=False, limit=3)
        lines.append(f"### {year}")
        lines.append("")
        lines.append("No-Reversal-No-Aux 更强的月份：")
        if best_alt:
            for row in best_alt:
                lines.append(
                    f"- `{row['month']}`: diff `{row['diff_r']:.4f}R` "
                    f"(No-Reversal-No-Aux `{row['no_reversal_no_aux_pnl_r']:.4f}R` vs Mainline `{row['mainline_pnl_r']:.4f}R`)"
                )
        else:
            lines.append("- 无")
        lines.append("")
        lines.append("Mainline 更强的月份：")
        if best_main:
            for row in best_main:
                lines.append(
                    f"- `{row['month']}`: diff `{row['diff_r']:.4f}R` "
                    f"(No-Reversal-No-Aux `{row['no_reversal_no_aux_pnl_r']:.4f}R` vs Mainline `{row['mainline_pnl_r']:.4f}R`)"
                )
        else:
            lines.append("- 无")
        lines.append("")
    return "\n".join(lines).strip()


def build_report(
    *,
    summary_rows: list[dict[str, object]],
    monthly_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    unmatched_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Mainline vs No-Reversal-No-Aux Trade Diff (2022 / 2024)",
        "",
        "## 目的",
        "",
        "- 不是重新评估整条策略家族，而是把 `Mainline (R1 RF1 HS1 AUX1)` 和 `No-Reversal-No-Aux (R0 RF1 HS1 AUX0)` 在 `2022`、`2024` 的逐笔 OOS 交易拆开。",
        "- 这里能回答的是“哪些交易两边都做了、哪些交易只有一边做”，不能直接证明“被过滤掉的信号本来一定会盈利”。",
        "- 为了避免简单最近邻错配，这里用的是同方向、按时间顺序的最大匹配，再在最大匹配数量下最小化总时间差；`exact` 表示同一 `signal_time`，`near` 表示 `12h` 内的近似同一 setup。",
        "",
        "## 总结",
        "",
        "| Year | Side | Mainline Trades | No-Rev-No-Aux Trades | Exact | Near | Mainline Only | No-Rev-No-Aux Only | Matched Delta R | Mainline Only R | No-Rev-No-Aux Only R | Overall Delta R |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(format_summary_row(row) for row in summary_rows)
    lines.extend(
        [
            "",
            "解读：",
            "- `Matched Delta R` 代表两边大致做的是同一批 setup，但收益不同。",
            "- `Mainline Only R` 和 `No-Rev-No-Aux Only R` 代表真正的交易选择差异，也就是一边做了、另一边没做。",
            "",
            "## 月度阶段",
            "",
            format_month_examples(monthly_rows),
        ]
    )

    for year in YEARS:
        year_pairs = [row for row in pair_rows if row["year"] == year]
        year_unmatched = [row for row in unmatched_rows if row["year"] == year]
        positive_pairs = top_rows(year_pairs, key="delta_r", reverse=True, limit=5)
        negative_pairs = top_rows(year_pairs, key="delta_r", reverse=False, limit=5)
        alt_only_winners = top_rows(
            [row for row in year_unmatched if row["origin"] == "no_reversal_no_aux_only"],
            key="pnl_r",
            reverse=True,
            limit=5,
        )
        main_only_winners = top_rows(
            [row for row in year_unmatched if row["origin"] == "mainline_only"],
            key="pnl_r",
            reverse=True,
            limit=5,
        )
        lines.extend(
            [
                "",
                f"## {year} Representative Trades",
                "",
                "No-Reversal-No-Aux 相比 Mainline 改善最大的近似同 setup：",
                markdown_table(
                    positive_pairs,
                    [
                        ("side", "Side"),
                        ("match_kind", "Kind"),
                        ("gap_hours", "Gap H"),
                        ("main_signal_time", "Main Signal"),
                        ("alt_signal_time", "Alt Signal"),
                        ("main_pnl_r", "Main R"),
                        ("alt_pnl_r", "Alt R"),
                        ("delta_r", "Delta R"),
                        ("main_exit_reason", "Main Exit"),
                        ("alt_exit_reason", "Alt Exit"),
                    ],
                ),
                "",
                "Mainline 相比 No-Reversal-No-Aux 改善最大的近似同 setup：",
                markdown_table(
                    negative_pairs,
                    [
                        ("side", "Side"),
                        ("match_kind", "Kind"),
                        ("gap_hours", "Gap H"),
                        ("main_signal_time", "Main Signal"),
                        ("alt_signal_time", "Alt Signal"),
                        ("main_pnl_r", "Main R"),
                        ("alt_pnl_r", "Alt R"),
                        ("delta_r", "Delta R"),
                        ("main_exit_reason", "Main Exit"),
                        ("alt_exit_reason", "Alt Exit"),
                    ],
                ),
                "",
                "只被 No-Reversal-No-Aux 放行且表现最好的交易：",
                markdown_table(
                    alt_only_winners,
                    [
                        ("side", "Side"),
                        ("signal_time", "Signal Time"),
                        ("pnl_r", "PnL R"),
                        ("exit_reason", "Exit"),
                        ("trend_strength", "Trend"),
                        ("confidence", "Conf"),
                    ],
                ),
                "",
                "只被 Mainline 放行且表现最好的交易：",
                markdown_table(
                    main_only_winners,
                    [
                        ("side", "Side"),
                        ("signal_time", "Signal Time"),
                        ("pnl_r", "PnL R"),
                        ("exit_reason", "Exit"),
                        ("trend_strength", "Trend"),
                        ("confidence", "Conf"),
                    ],
                ),
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ensure_output_dir()
    trades = load_trades()

    all_pair_rows: list[dict[str, object]] = []
    all_unmatched_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for year in YEARS:
        for side in sorted(trades.loc[trades["year"] == year, "side"].unique()):
            pair_rows, unmatched_rows, summary = summarize_pairs(trades=trades, year=year, side=side)
            all_pair_rows.extend(pair_rows)
            all_unmatched_rows.extend(unmatched_rows)
            summary_rows.append(summary)

    monthly_rows = build_monthly_rows(trades)

    write_csv(OUTPUT_DIR / "matched_pairs.csv", all_pair_rows)
    write_csv(OUTPUT_DIR / "unmatched_trades.csv", all_unmatched_rows)
    write_csv(OUTPUT_DIR / "match_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "monthly_diff.csv", monthly_rows)

    report = build_report(
        summary_rows=summary_rows,
        monthly_rows=monthly_rows,
        pair_rows=all_pair_rows,
        unmatched_rows=all_unmatched_rows,
    )
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(f"Saved report: {OUTPUT_DIR / 'report.md'}")
    print(f"Saved matched pairs CSV: {OUTPUT_DIR / 'matched_pairs.csv'}")
    print(f"Saved unmatched trades CSV: {OUTPUT_DIR / 'unmatched_trades.csv'}")
    print(f"Saved summary CSV: {OUTPUT_DIR / 'match_summary.csv'}")
    print(f"Saved monthly diff CSV: {OUTPUT_DIR / 'monthly_diff.csv'}")


if __name__ == "__main__":
    main()
