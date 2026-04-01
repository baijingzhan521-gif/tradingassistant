from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts/backtests/post_tp1_extension_mainline"
OUTPUT_DIR = ROOT / "artifacts/backtests/post_tp1_extension_diff"

WINDOWS = ("two_year", "full_2020")
BASELINE_PROFILE = "baseline_be_after_tp1"
EXTENSION_PROFILE = "be_if_no_extension_within_3bars_after_tp1"

PROFILE_LABELS = {
    BASELINE_PROFILE: "Baseline: BE After TP1",
    EXTENSION_PROFILE: "3 Bars: BE If No 1H Extension After TP1",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_trades(window: str, profile: str) -> pd.DataFrame:
    path = SOURCE_DIR / f"{window}_{profile}_trades.csv"
    frame = pd.read_csv(path, parse_dates=["signal_time", "entry_time", "exit_time"])
    frame = frame[frame["side"] == "LONG"].copy()
    frame["year"] = frame["entry_time"].dt.year
    frame["month"] = frame["entry_time"].dt.strftime("%Y-%m")
    frame["profile_key"] = profile
    frame["profile_label"] = PROFILE_LABELS[profile]
    return frame.sort_values("entry_time").reset_index(drop=True)


def build_match_key(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.copy()
    keyed["match_key"] = (
        keyed["signal_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        + "|"
        + keyed["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        + "|"
        + keyed["side"]
    )
    return keyed


def find_blocker(row: pd.Series, other_frame: pd.DataFrame) -> pd.Series | None:
    overlaps = other_frame[
        (other_frame["entry_time"] <= row["entry_time"]) & (other_frame["exit_time"] >= row["entry_time"])
    ]
    if overlaps.empty:
        return None
    return overlaps.sort_values("entry_time").iloc[0]


def summarize_window(
    window: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    baseline = build_match_key(load_trades(window, BASELINE_PROFILE))
    extension = build_match_key(load_trades(window, EXTENSION_PROFILE))
    key = ["match_key"]

    matched = baseline.merge(
        extension,
        on=key,
        suffixes=("_baseline", "_ext"),
        how="inner",
    )
    matched["delta_r"] = matched["pnl_r_ext"] - matched["pnl_r_baseline"]

    matched_rows: list[dict[str, Any]] = []
    for row in matched.itertuples(index=False):
        matched_rows.append(
            {
                "window": window,
                "year": int(row.year_baseline),
                "signal_time": row.signal_time_baseline.isoformat(),
                "entry_time": row.entry_time_baseline.isoformat(),
                "baseline_exit_time": row.exit_time_baseline.isoformat(),
                "extension_exit_time": row.exit_time_ext.isoformat(),
                "trend_strength": int(row.trend_strength_baseline),
                "baseline_exit_reason": row.exit_reason_baseline,
                "extension_exit_reason": row.exit_reason_ext,
                "baseline_bars_held": int(row.bars_held_baseline),
                "extension_bars_held": int(row.bars_held_ext),
                "baseline_pnl_r": round(float(row.pnl_r_baseline), 4),
                "extension_pnl_r": round(float(row.pnl_r_ext), 4),
                "delta_r": round(float(row.delta_r), 4),
                "extension_decision_made": bool(row.extension_decision_made_ext)
                if pd.notna(row.extension_decision_made_ext)
                else False,
                "extension_confirmed": bool(row.extension_confirmed_ext)
                if pd.notna(row.extension_confirmed_ext)
                else False,
                "extension_decision_reason": row.extension_decision_reason_ext
                if pd.notna(row.extension_decision_reason_ext)
                else "",
                "extension_observed_bars": int(row.extension_observed_bars_ext)
                if pd.notna(row.extension_observed_bars_ext)
                else 0,
            }
        )

    baseline_only = baseline[~baseline["match_key"].isin(matched["match_key"])].copy()
    extension_only = extension[~extension["match_key"].isin(matched["match_key"])].copy()

    unmatched_rows: list[dict[str, Any]] = []
    for origin, frame, other_frame in (
        ("baseline_only", baseline_only, extension),
        ("extension_only", extension_only, baseline),
    ):
        for row in frame.itertuples(index=False):
            blocker = find_blocker(pd.Series(row._asdict()), other_frame)
            unmatched_rows.append(
                {
                    "window": window,
                    "year": int(row.year),
                    "origin": origin,
                    "signal_time": row.signal_time.isoformat(),
                    "entry_time": row.entry_time.isoformat(),
                    "exit_time": row.exit_time.isoformat(),
                    "trend_strength": int(row.trend_strength),
                    "exit_reason": row.exit_reason,
                    "bars_held": int(row.bars_held),
                    "pnl_r": round(float(row.pnl_r), 4),
                    "blocked_by_open_position": blocker is not None,
                    "blocked_by_signal_time": blocker["signal_time"].isoformat() if blocker is not None else "",
                    "blocked_by_entry_time": blocker["entry_time"].isoformat() if blocker is not None else "",
                    "blocked_by_exit_time": blocker["exit_time"].isoformat() if blocker is not None else "",
                    "blocked_by_exit_reason": str(blocker["exit_reason"]) if blocker is not None else "",
                    "blocked_by_pnl_r": round(float(blocker["pnl_r"]), 4) if blocker is not None else 0.0,
                }
            )

    transition = (
        matched.groupby(["exit_reason_baseline", "exit_reason_ext"])["delta_r"]
        .agg(["count", "sum"])
        .reset_index()
        .sort_values(["sum", "count"], ascending=[False, False])
    )
    transition_rows: list[dict[str, Any]] = []
    for row in transition.itertuples(index=False):
        transition_rows.append(
            {
                "window": window,
                "transition": f"{row.exit_reason_baseline} -> {row.exit_reason_ext}",
                "count": int(row.count),
                "delta_r_sum": round(float(row.sum), 4),
            }
        )

    yearly_rows: list[dict[str, Any]] = []
    for year in sorted(set(baseline["year"]).union(set(extension["year"]))):
        matched_year = matched[matched["year_baseline"] == year]
        baseline_only_year = baseline_only[baseline_only["year"] == year]
        extension_only_year = extension_only[extension_only["year"] == year]
        yearly_rows.append(
            {
                "window": window,
                "year": int(year),
                "matched_trade_count": int(len(matched_year)),
                "matched_delta_r": round(float(matched_year["delta_r"].sum()), 4),
                "baseline_only_count": int(len(baseline_only_year)),
                "baseline_only_pnl_r": round(float(baseline_only_year["pnl_r"].sum()), 4),
                "extension_only_count": int(len(extension_only_year)),
                "extension_only_pnl_r": round(float(extension_only_year["pnl_r"].sum()), 4),
                "overall_delta_r": round(
                    float(matched_year["delta_r"].sum() + extension_only_year["pnl_r"].sum() - baseline_only_year["pnl_r"].sum()),
                    4,
                ),
            }
        )

    summary = {
        "window": window,
        "baseline_trades": int(len(baseline)),
        "extension_trades": int(len(extension)),
        "matched_trades": int(len(matched)),
        "baseline_only": int(len(baseline_only)),
        "extension_only": int(len(extension_only)),
        "matched_delta_r": round(float(matched["delta_r"].sum()), 4),
        "baseline_only_pnl_r": round(float(baseline_only["pnl_r"].sum()), 4),
        "extension_only_pnl_r": round(float(extension_only["pnl_r"].sum()), 4),
        "overall_delta_r": round(float(extension["pnl_r"].sum() - baseline["pnl_r"].sum()), 4),
        "baseline_only_blocked_count": int(sum(bool(row["blocked_by_open_position"]) for row in unmatched_rows if row["origin"] == "baseline_only")),
        "extension_only_blocked_count": int(sum(bool(row["blocked_by_open_position"]) for row in unmatched_rows if row["origin"] == "extension_only")),
    }
    return matched_rows, unmatched_rows, transition_rows, yearly_rows, summary


def top_rows(rows: list[dict[str, Any]], *, key: str, reverse: bool, limit: int = 8) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: item[key], reverse=reverse)[:limit]


def build_report(
    *,
    summaries: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    yearly_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Baseline vs 3-Bar Extension LONG Diff",
        "",
        "- 比较对象只限 `LONG`，因为这轮 post-TP1 规则只改 `LONG`。",
        "- `matched` 指完全相同的 `signal_time + entry_time + side` 共享交易。",
        "- `baseline_only / extension_only` 不是新信号源差异，而是 sequence-aware 下持仓时长变化导致后续 `LONG` 被跳过或释放。",
        "",
        "## Summary",
        "",
        markdown_table(
            summaries,
            [
                ("window", "Window"),
                ("baseline_trades", "Baseline Trades"),
                ("extension_trades", "3-Bar Trades"),
                ("matched_trades", "Matched"),
                ("baseline_only", "Baseline Only"),
                ("extension_only", "3-Bar Only"),
                ("matched_delta_r", "Matched Delta R"),
                ("baseline_only_pnl_r", "Baseline Only R"),
                ("extension_only_pnl_r", "3-Bar Only R"),
                ("overall_delta_r", "Overall Delta R"),
            ],
        ),
        "",
    ]

    for summary in summaries:
        window = summary["window"]
        lines.extend(
            [
                f"## {window}",
                "",
                f"- 共享交易净变化：`{summary['matched_delta_r']:+.4f}R`",
                f"- `baseline_only`：`{summary['baseline_only']} 笔 / {summary['baseline_only_pnl_r']:+.4f}R`，其中被 3-bar 持仓直接卡掉的有 `{summary['baseline_only_blocked_count']}` 笔。",
                f"- `3-bar only`：`{summary['extension_only']} 笔 / {summary['extension_only_pnl_r']:+.4f}R`，其中被 baseline 持仓卡掉的有 `{summary['extension_only_blocked_count']}` 笔。",
                f"- 最终 long 侧总差值：`{summary['overall_delta_r']:+.4f}R`。",
                "",
                "转移矩阵：",
                "",
                markdown_table(
                    [row for row in transitions if row["window"] == window],
                    [("transition", "Transition"), ("count", "Count"), ("delta_r_sum", "Delta R")],
                ),
                "",
                "年度拆解：",
                "",
                markdown_table(
                    [row for row in yearly_rows if row["window"] == window],
                    [
                        ("year", "Year"),
                        ("matched_trade_count", "Matched"),
                        ("matched_delta_r", "Matched Delta R"),
                        ("baseline_only_count", "Baseline Only"),
                        ("baseline_only_pnl_r", "Baseline Only R"),
                        ("extension_only_count", "3-Bar Only"),
                        ("extension_only_pnl_r", "3-Bar Only R"),
                        ("overall_delta_r", "Overall Delta R"),
                    ],
                ),
                "",
                "共享交易里 3-bar 最有利的样本：",
                "",
                markdown_table(
                    top_rows([row for row in matched_rows if row["window"] == window], key="delta_r", reverse=True),
                    [
                        ("signal_time", "Signal"),
                        ("baseline_exit_reason", "Baseline Exit"),
                        ("extension_exit_reason", "3-Bar Exit"),
                        ("baseline_pnl_r", "Baseline R"),
                        ("extension_pnl_r", "3-Bar R"),
                        ("delta_r", "Delta R"),
                        ("extension_decision_reason", "3-Bar Decision"),
                    ],
                ),
                "",
                "共享交易里 3-bar 最不利的样本：",
                "",
                markdown_table(
                    top_rows([row for row in matched_rows if row["window"] == window], key="delta_r", reverse=False),
                    [
                        ("signal_time", "Signal"),
                        ("baseline_exit_reason", "Baseline Exit"),
                        ("extension_exit_reason", "3-Bar Exit"),
                        ("baseline_pnl_r", "Baseline R"),
                        ("extension_pnl_r", "3-Bar R"),
                        ("delta_r", "Delta R"),
                        ("extension_decision_reason", "3-Bar Decision"),
                    ],
                ),
                "",
                "被持仓时长卡掉的主要样本：",
                "",
                markdown_table(
                    top_rows(
                        [row for row in unmatched_rows if row["window"] == window and row["origin"] == "baseline_only"],
                        key="pnl_r",
                        reverse=True,
                    ),
                    [
                        ("signal_time", "Signal"),
                        ("exit_reason", "Baseline Exit"),
                        ("pnl_r", "Baseline R"),
                        ("blocked_by_signal_time", "Blocked By"),
                        ("blocked_by_exit_reason", "3-Bar Exit"),
                        ("blocked_by_pnl_r", "3-Bar R"),
                    ],
                ),
                "",
            ]
        )
        ext_only_rows = [row for row in unmatched_rows if row["window"] == window and row["origin"] == "extension_only"]
        if ext_only_rows:
            lines.extend(
                [
                    "3-bar 独有样本：",
                    "",
                    markdown_table(
                        ext_only_rows,
                        [
                            ("signal_time", "Signal"),
                            ("exit_reason", "3-Bar Exit"),
                            ("pnl_r", "3-Bar R"),
                            ("blocked_by_signal_time", "Blocked By"),
                            ("blocked_by_exit_reason", "Baseline Exit"),
                            ("blocked_by_pnl_r", "Baseline R"),
                        ],
                    ),
                    "",
                ]
            )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ensure_output_dir()

    all_matched_rows: list[dict[str, Any]] = []
    all_unmatched_rows: list[dict[str, Any]] = []
    all_transition_rows: list[dict[str, Any]] = []
    all_yearly_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for window in WINDOWS:
        matched_rows, unmatched_rows, transition_rows, yearly_rows, summary = summarize_window(window)
        all_matched_rows.extend(matched_rows)
        all_unmatched_rows.extend(unmatched_rows)
        all_transition_rows.extend(transition_rows)
        all_yearly_rows.extend(yearly_rows)
        summaries.append(summary)

    write_csv(OUTPUT_DIR / "matched_pairs.csv", all_matched_rows)
    write_csv(OUTPUT_DIR / "unmatched_trades.csv", all_unmatched_rows)
    write_csv(OUTPUT_DIR / "transition_summary.csv", all_transition_rows)
    write_csv(OUTPUT_DIR / "yearly_breakdown.csv", all_yearly_rows)
    write_csv(OUTPUT_DIR / "summary.csv", summaries)

    report = build_report(
        summaries=summaries,
        transitions=all_transition_rows,
        yearly_rows=all_yearly_rows,
        matched_rows=all_matched_rows,
        unmatched_rows=all_unmatched_rows,
    )
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved report: {report_path}")
    print(f"Saved matched pairs CSV: {OUTPUT_DIR / 'matched_pairs.csv'}")
    print(f"Saved unmatched trades CSV: {OUTPUT_DIR / 'unmatched_trades.csv'}")
    print(f"Saved transition summary CSV: {OUTPUT_DIR / 'transition_summary.csv'}")
    print(f"Saved yearly breakdown CSV: {OUTPUT_DIR / 'yearly_breakdown.csv'}")
    print(f"Saved summary CSV: {OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
