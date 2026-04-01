from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts/backtests/post_tp1_extension_diff"
OUTPUT_DIR = ROOT / "artifacts/backtests/post_tp1_chain_state_study"

WINDOWS = ("two_year", "full_2020")


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


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    matched = pd.read_csv(
        SOURCE_DIR / "matched_pairs.csv",
        parse_dates=["signal_time", "entry_time", "baseline_exit_time", "extension_exit_time"],
    )
    unmatched = pd.read_csv(
        SOURCE_DIR / "unmatched_trades.csv",
        parse_dates=[
            "signal_time",
            "entry_time",
            "exit_time",
            "blocked_by_signal_time",
            "blocked_by_entry_time",
            "blocked_by_exit_time",
        ],
    )
    return matched, unmatched


def bucket_trend_strength(value: int) -> str:
    if value <= 92:
        return "<=92"
    if value <= 95:
        return "93-95"
    if value <= 98:
        return "96-98"
    return "99+"


def bucket_extra_bars(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 6:
        return "1-6"
    if value <= 24:
        return "7-24"
    if value <= 72:
        return "25-72"
    return "73+"


def bucket_blocked_count(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    return "2+"


def normalize_decision(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "none"
    return str(value)


def build_chain_rows(
    *,
    matched: pd.DataFrame,
    unmatched: pd.DataFrame,
    window: str,
) -> list[dict[str, Any]]:
    matched = matched[matched["window"] == window].copy()
    unmatched = unmatched[(unmatched["window"] == window) & (unmatched["origin"] == "baseline_only")].copy()

    rows: list[dict[str, Any]] = []
    for row in matched.itertuples(index=False):
        blocked = unmatched[unmatched["blocked_by_signal_time"] == row.signal_time].copy()
        extra_bars = int(row.extension_bars_held - row.baseline_bars_held)
        blocked_total_r = float(blocked["pnl_r"].sum())
        chain_net_r = float(row.delta_r - blocked_total_r)
        rows.append(
            {
                "window": window,
                "signal_time": row.signal_time.isoformat(),
                "entry_time": row.entry_time.isoformat(),
                "year": int(row.year),
                "month": row.entry_time.strftime("%Y-%m"),
                "trend_strength": int(row.trend_strength),
                "trend_bucket": bucket_trend_strength(int(row.trend_strength)),
                "baseline_exit_reason": row.baseline_exit_reason,
                "extension_exit_reason": row.extension_exit_reason,
                "delta_r": round(float(row.delta_r), 4),
                "baseline_bars_held": int(row.baseline_bars_held),
                "extension_bars_held": int(row.extension_bars_held),
                "extra_bars": extra_bars,
                "extra_bars_bucket": bucket_extra_bars(extra_bars),
                "decision_reason": normalize_decision(row.extension_decision_reason),
                "blocked_count": int(len(blocked)),
                "blocked_count_bucket": bucket_blocked_count(int(len(blocked))),
                "blocked_total_r": round(blocked_total_r, 4),
                "chain_net_r": round(chain_net_r, 4),
                "net_positive": chain_net_r > 0,
                "net_negative": chain_net_r < 0,
                "blocked_positive_r": round(float(blocked[blocked["pnl_r"] > 0]["pnl_r"].sum()), 4),
                "blocked_negative_r": round(float(blocked[blocked["pnl_r"] < 0]["pnl_r"].sum()), 4),
            }
        )
    return rows


def aggregate_rows(
    chain_df: pd.DataFrame,
    *,
    group_field: str,
) -> list[dict[str, Any]]:
    grouped = (
        chain_df.groupby(group_field, observed=False)
        .agg(
            chains=("signal_time", "size"),
            positive_chains=("net_positive", "sum"),
            negative_chains=("net_negative", "sum"),
            delta_r_sum=("delta_r", "sum"),
            blocked_total_r_sum=("blocked_total_r", "sum"),
            chain_net_r_sum=("chain_net_r", "sum"),
            chain_net_r_mean=("chain_net_r", "mean"),
            blocked_count_sum=("blocked_count", "sum"),
            extra_bars_mean=("extra_bars", "mean"),
        )
        .reset_index()
        .sort_values("chain_net_r_sum", ascending=False)
    )
    rows: list[dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "group": getattr(row, group_field),
                "chains": int(row.chains),
                "positive_chains": int(row.positive_chains),
                "negative_chains": int(row.negative_chains),
                "delta_r_sum": round(float(row.delta_r_sum), 4),
                "blocked_total_r_sum": round(float(row.blocked_total_r_sum), 4),
                "chain_net_r_sum": round(float(row.chain_net_r_sum), 4),
                "chain_net_r_mean": round(float(row.chain_net_r_mean), 4),
                "blocked_count_sum": int(row.blocked_count_sum),
                "extra_bars_mean": round(float(row.extra_bars_mean), 2),
            }
        )
    return rows


def build_report(
    *,
    summaries: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    trend_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
    best_chains: list[dict[str, Any]],
    worst_chains: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Post-TP1 Chain State Study",
        "",
        "- `chain` 的定义是：同一个 3-bar 锚点单的 `anchor delta R`，减去它因为持仓延长而卡掉的后续 baseline LONG 总收益。",
        "- 这一步只是在寻找模式，不是假装已经找到了可部署的启停规则。",
        "- 需要特别注意：`blocked_count` 和 `blocked_total_r` 都是事后量，不能直接拿来当线上规则。",
        "",
        "## Summary",
        "",
        markdown_table(
            summaries,
            [
                ("window", "Window"),
                ("chains", "Chains"),
                ("positive_chains", "Positive"),
                ("negative_chains", "Negative"),
                ("delta_r_sum", "Anchor Delta R"),
                ("blocked_total_r_sum", "Blocked Cost R"),
                ("chain_net_r_sum", "Chain Net R"),
                ("chain_net_r_mean", "Chain Mean R"),
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
                f"- 总锚点链条：`{summary['chains']}`，净正链条 `{summary['positive_chains']}`，净负链条 `{summary['negative_chains']}`。",
                f"- 锚点单本身累计贡献：`{summary['delta_r_sum']:+.4f}R`。",
                f"- 被卡掉后续 baseline LONG 的累计收益：`{summary['blocked_total_r_sum']:+.4f}R`。",
                f"- 链条净值：`{summary['chain_net_r_sum']:+.4f}R`，平均每条 `{summary['chain_net_r_mean']:+.4f}R`。",
                "",
                "按月份：",
                "",
                markdown_table(
                    [row for row in monthly_rows if row["window"] == window][:12],
                    [
                        ("group", "Month"),
                        ("chains", "Chains"),
                        ("delta_r_sum", "Anchor Delta R"),
                        ("blocked_total_r_sum", "Blocked Cost R"),
                        ("chain_net_r_sum", "Chain Net R"),
                        ("positive_chains", "Positive"),
                        ("negative_chains", "Negative"),
                    ],
                ),
                "",
                "按 trend_strength 分桶：",
                "",
                markdown_table(
                    [row for row in trend_rows if row["window"] == window],
                    [
                        ("group", "Trend Bucket"),
                        ("chains", "Chains"),
                        ("chain_net_r_sum", "Chain Net R"),
                        ("chain_net_r_mean", "Mean R"),
                        ("delta_r_sum", "Anchor Delta R"),
                        ("blocked_total_r_sum", "Blocked Cost R"),
                    ],
                ),
                "",
                "按 TP1 后 decision：",
                "",
                markdown_table(
                    [row for row in decision_rows if row["window"] == window],
                    [
                        ("group", "Decision"),
                        ("chains", "Chains"),
                        ("chain_net_r_sum", "Chain Net R"),
                        ("chain_net_r_mean", "Mean R"),
                        ("blocked_total_r_sum", "Blocked Cost R"),
                    ],
                ),
                "",
                "按 blocked_count 分桶：",
                "",
                markdown_table(
                    [row for row in blocked_rows if row["window"] == window],
                    [
                        ("group", "Blocked Count"),
                        ("chains", "Chains"),
                        ("chain_net_r_sum", "Chain Net R"),
                        ("chain_net_r_mean", "Mean R"),
                        ("delta_r_sum", "Anchor Delta R"),
                        ("blocked_total_r_sum", "Blocked Cost R"),
                    ],
                ),
                "",
                "按 extra_bars 分桶：",
                "",
                markdown_table(
                    [row for row in extra_rows if row["window"] == window],
                    [
                        ("group", "Extra Bars"),
                        ("chains", "Chains"),
                        ("chain_net_r_sum", "Chain Net R"),
                        ("chain_net_r_mean", "Mean R"),
                        ("blocked_total_r_sum", "Blocked Cost R"),
                    ],
                ),
                "",
                "最好的链条：",
                "",
                markdown_table(
                    [row for row in best_chains if row["window"] == window],
                    [
                        ("signal_time", "Signal"),
                        ("month", "Month"),
                        ("trend_strength", "Trend"),
                        ("delta_r", "Anchor Delta R"),
                        ("blocked_count", "Blocked"),
                        ("blocked_total_r", "Blocked Cost R"),
                        ("chain_net_r", "Chain Net R"),
                        ("decision_reason", "Decision"),
                    ],
                ),
                "",
                "最差的链条：",
                "",
                markdown_table(
                    [row for row in worst_chains if row["window"] == window],
                    [
                        ("signal_time", "Signal"),
                        ("month", "Month"),
                        ("trend_strength", "Trend"),
                        ("delta_r", "Anchor Delta R"),
                        ("blocked_count", "Blocked"),
                        ("blocked_total_r", "Blocked Cost R"),
                        ("chain_net_r", "Chain Net R"),
                        ("decision_reason", "Decision"),
                    ],
                ),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ensure_output_dir()
    matched, unmatched = load_frames()

    chain_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    extra_rows: list[dict[str, Any]] = []
    best_chains: list[dict[str, Any]] = []
    worst_chains: list[dict[str, Any]] = []

    for window in WINDOWS:
        rows = build_chain_rows(matched=matched, unmatched=unmatched, window=window)
        chain_rows.extend(rows)
        chain_df = pd.DataFrame(rows)

        summaries.append(
            {
                "window": window,
                "chains": int(len(chain_df)),
                "positive_chains": int(chain_df["net_positive"].sum()),
                "negative_chains": int(chain_df["net_negative"].sum()),
                "delta_r_sum": round(float(chain_df["delta_r"].sum()), 4),
                "blocked_total_r_sum": round(float(chain_df["blocked_total_r"].sum()), 4),
                "chain_net_r_sum": round(float(chain_df["chain_net_r"].sum()), 4),
                "chain_net_r_mean": round(float(chain_df["chain_net_r"].mean()), 4),
            }
        )

        monthly_rows.extend({"window": window, **row} for row in aggregate_rows(chain_df, group_field="month"))
        trend_rows.extend({"window": window, **row} for row in aggregate_rows(chain_df, group_field="trend_bucket"))
        decision_rows.extend({"window": window, **row} for row in aggregate_rows(chain_df, group_field="decision_reason"))
        blocked_rows.extend({"window": window, **row} for row in aggregate_rows(chain_df, group_field="blocked_count_bucket"))
        extra_rows.extend({"window": window, **row} for row in aggregate_rows(chain_df, group_field="extra_bars_bucket"))

        top_pos = chain_df.sort_values("chain_net_r", ascending=False).head(10)
        top_neg = chain_df.sort_values("chain_net_r").head(10)
        best_chains.extend(top_pos.to_dict(orient="records"))
        worst_chains.extend(top_neg.to_dict(orient="records"))

    write_csv(OUTPUT_DIR / "chain_rows.csv", chain_rows)
    write_csv(OUTPUT_DIR / "summary.csv", summaries)
    write_csv(OUTPUT_DIR / "monthly_breakdown.csv", monthly_rows)
    write_csv(OUTPUT_DIR / "trend_breakdown.csv", trend_rows)
    write_csv(OUTPUT_DIR / "decision_breakdown.csv", decision_rows)
    write_csv(OUTPUT_DIR / "blocked_count_breakdown.csv", blocked_rows)
    write_csv(OUTPUT_DIR / "extra_bars_breakdown.csv", extra_rows)

    report = build_report(
        summaries=summaries,
        monthly_rows=monthly_rows,
        trend_rows=trend_rows,
        decision_rows=decision_rows,
        blocked_rows=blocked_rows,
        extra_rows=extra_rows,
        best_chains=best_chains,
        worst_chains=worst_chains,
    )
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved report: {report_path}")
    print(f"Saved chain rows CSV: {OUTPUT_DIR / 'chain_rows.csv'}")
    print(f"Saved summary CSV: {OUTPUT_DIR / 'summary.csv'}")
    print(f"Saved monthly breakdown CSV: {OUTPUT_DIR / 'monthly_breakdown.csv'}")
    print(f"Saved trend breakdown CSV: {OUTPUT_DIR / 'trend_breakdown.csv'}")
    print(f"Saved decision breakdown CSV: {OUTPUT_DIR / 'decision_breakdown.csv'}")
    print(f"Saved blocked count breakdown CSV: {OUTPUT_DIR / 'blocked_count_breakdown.csv'}")
    print(f"Saved extra bars breakdown CSV: {OUTPUT_DIR / 'extra_bars_breakdown.csv'}")


if __name__ == "__main__":
    main()
