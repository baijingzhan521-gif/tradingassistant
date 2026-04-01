from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts/backtests/post_tp1_extension_trend_veto_mainline"
OUTPUT_DIR = ROOT / "artifacts/backtests/post_tp1_extension_veto_vs_threebars_diff"

WINDOWS = ("two_year", "full_2020")
ORIGINAL_PROFILE = "be_if_no_extension_within_3bars_after_tp1"
VETO_PROFILE = "be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98"

PROFILE_LABELS = {
    ORIGINAL_PROFILE: "3 Bars: BE If No 1H Extension After TP1",
    VETO_PROFILE: "3 Bars + Trend 96-98 Veto",
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
    frame["match_key"] = (
        frame["signal_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        + "|"
        + frame["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        + "|"
        + frame["side"]
    )
    return frame.sort_values("entry_time").reset_index(drop=True)


def find_blocker(row: pd.Series, other_frame: pd.DataFrame) -> pd.Series | None:
    overlaps = other_frame[
        (other_frame["entry_time"] <= row["entry_time"]) & (other_frame["exit_time"] >= row["entry_time"])
    ]
    if overlaps.empty:
        return None
    return overlaps.sort_values("entry_time").iloc[0]


def build_chain_rows(
    *,
    window: str,
    matched: pd.DataFrame,
    unmatched_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched_by_key = {
        row["match_key"]: row
        for row in matched[
            [
                "match_key",
                "signal_time_orig",
                "entry_time_orig",
                "trend_strength_orig",
                "exit_reason_orig",
                "exit_reason_veto",
                "pnl_r_orig",
                "pnl_r_veto",
                "delta_r",
                "extension_decision_reason_orig",
                "extension_decision_reason_veto",
            ]
        ].to_dict("records")
    }

    chain_map: dict[str, dict[str, Any]] = {}

    def ensure_chain(anchor_key: str) -> dict[str, Any]:
        if anchor_key not in chain_map:
            anchor = matched_by_key.get(anchor_key)
            if anchor is None:
                raise KeyError(f"Missing matched anchor for {anchor_key}")
            chain_map[anchor_key] = {
                "window": window,
                "year": pd.Timestamp(anchor["entry_time_orig"]).year,
                "month": pd.Timestamp(anchor["entry_time_orig"]).strftime("%Y-%m"),
                "anchor_match_key": anchor_key,
                "anchor_signal_time": pd.Timestamp(anchor["signal_time_orig"]).isoformat(),
                "anchor_entry_time": pd.Timestamp(anchor["entry_time_orig"]).isoformat(),
                "trend_strength": int(anchor["trend_strength_orig"]),
                "original_exit_reason": anchor["exit_reason_orig"],
                "veto_exit_reason": anchor["exit_reason_veto"],
                "original_pnl_r": round(float(anchor["pnl_r_orig"]), 4),
                "veto_pnl_r": round(float(anchor["pnl_r_veto"]), 4),
                "anchor_delta_r": round(float(anchor["delta_r"]), 4),
                "original_decision_reason": anchor["extension_decision_reason_orig"]
                if pd.notna(anchor["extension_decision_reason_orig"])
                else "",
                "veto_decision_reason": anchor["extension_decision_reason_veto"]
                if pd.notna(anchor["extension_decision_reason_veto"])
                else "",
                "released_trade_count": 0,
                "released_total_r": 0.0,
                "blocked_trade_count": 0,
                "blocked_total_r": 0.0,
            }
        return chain_map[anchor_key]

    # Start with matched anchors that actually changed.
    for row in matched.itertuples(index=False):
        if abs(float(row.delta_r)) > 1e-9:
            ensure_chain(row.match_key)

    for row in unmatched_rows:
        if not row["blocked_by_open_position"]:
            continue
        anchor_key = row["blocked_by_match_key"]
        if not anchor_key:
            continue
        chain = ensure_chain(anchor_key)
        if row["origin"] == "veto_only":
            chain["released_trade_count"] += 1
            chain["released_total_r"] += float(row["pnl_r"])
        elif row["origin"] == "original_only":
            chain["blocked_trade_count"] += 1
            chain["blocked_total_r"] += float(row["pnl_r"])

    chain_rows: list[dict[str, Any]] = []
    for chain in chain_map.values():
        released_total_r = round(float(chain["released_total_r"]), 4)
        blocked_total_r = round(float(chain["blocked_total_r"]), 4)
        net_chain_delta_r = round(float(chain["anchor_delta_r"] + released_total_r - blocked_total_r), 4)
        chain_rows.append(
            {
                **chain,
                "released_total_r": released_total_r,
                "blocked_total_r": blocked_total_r,
                "net_chain_delta_r": net_chain_delta_r,
            }
        )

    return sorted(chain_rows, key=lambda item: (item["window"], item["anchor_entry_time"]))


def summarize_window(
    window: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    original = load_trades(window, ORIGINAL_PROFILE)
    veto = load_trades(window, VETO_PROFILE)

    matched = original.merge(
        veto,
        on=["match_key"],
        suffixes=("_orig", "_veto"),
        how="inner",
    )
    matched["delta_r"] = matched["pnl_r_veto"] - matched["pnl_r_orig"]

    matched_rows: list[dict[str, Any]] = []
    for row in matched.itertuples(index=False):
        matched_rows.append(
            {
                "window": window,
                "year": int(row.year_orig),
                "month": row.month_orig,
                "match_key": row.match_key,
                "signal_time": row.signal_time_orig.isoformat(),
                "entry_time": row.entry_time_orig.isoformat(),
                "original_exit_time": row.exit_time_orig.isoformat(),
                "veto_exit_time": row.exit_time_veto.isoformat(),
                "trend_strength": int(row.trend_strength_orig),
                "original_exit_reason": row.exit_reason_orig,
                "veto_exit_reason": row.exit_reason_veto,
                "original_bars_held": int(row.bars_held_orig),
                "veto_bars_held": int(row.bars_held_veto),
                "original_pnl_r": round(float(row.pnl_r_orig), 4),
                "veto_pnl_r": round(float(row.pnl_r_veto), 4),
                "delta_r": round(float(row.delta_r), 4),
                "original_decision_reason": row.extension_decision_reason_orig
                if pd.notna(row.extension_decision_reason_orig)
                else "",
                "veto_decision_reason": row.extension_decision_reason_veto
                if pd.notna(row.extension_decision_reason_veto)
                else "",
            }
        )

    original_only = original[~original["match_key"].isin(matched["match_key"])].copy()
    veto_only = veto[~veto["match_key"].isin(matched["match_key"])].copy()

    unmatched_rows: list[dict[str, Any]] = []
    for origin, frame, other_frame in (
        ("original_only", original_only, veto),
        ("veto_only", veto_only, original),
    ):
        for row in frame.itertuples(index=False):
            blocker = find_blocker(pd.Series(row._asdict()), other_frame)
            unmatched_rows.append(
                {
                    "window": window,
                    "year": int(row.year),
                    "month": row.month,
                    "origin": origin,
                    "signal_time": row.signal_time.isoformat(),
                    "entry_time": row.entry_time.isoformat(),
                    "exit_time": row.exit_time.isoformat(),
                    "match_key": row.match_key,
                    "trend_strength": int(row.trend_strength),
                    "exit_reason": row.exit_reason,
                    "bars_held": int(row.bars_held),
                    "pnl_r": round(float(row.pnl_r), 4),
                    "blocked_by_open_position": blocker is not None,
                    "blocked_by_match_key": str(blocker["match_key"]) if blocker is not None else "",
                    "blocked_by_signal_time": blocker["signal_time"].isoformat() if blocker is not None else "",
                    "blocked_by_entry_time": blocker["entry_time"].isoformat() if blocker is not None else "",
                    "blocked_by_exit_time": blocker["exit_time"].isoformat() if blocker is not None else "",
                    "blocked_by_exit_reason": str(blocker["exit_reason"]) if blocker is not None else "",
                    "blocked_by_pnl_r": round(float(blocker["pnl_r"]), 4) if blocker is not None else 0.0,
                }
            )

    monthly_rows: list[dict[str, Any]] = []
    months = sorted(set(original["month"]).union(set(veto["month"])))
    for month in months:
        matched_month = matched[matched["month_orig"] == month]
        original_only_month = original_only[original_only["month"] == month]
        veto_only_month = veto_only[veto_only["month"] == month]
        overall_delta_r = float(
            matched_month["delta_r"].sum() + veto_only_month["pnl_r"].sum() - original_only_month["pnl_r"].sum()
        )
        if abs(overall_delta_r) <= 1e-9:
            continue
        monthly_rows.append(
            {
                "window": window,
                "month": month,
                "matched_delta_r": round(float(matched_month["delta_r"].sum()), 4),
                "original_only_pnl_r": round(float(original_only_month["pnl_r"].sum()), 4),
                "veto_only_pnl_r": round(float(veto_only_month["pnl_r"].sum()), 4),
                "overall_delta_r": round(overall_delta_r, 4),
            }
        )

    chain_rows = build_chain_rows(window=window, matched=matched, unmatched_rows=unmatched_rows)

    summary = {
        "window": window,
        "original_trades": int(len(original)),
        "veto_trades": int(len(veto)),
        "matched_trades": int(len(matched)),
        "changed_matched_trades": int(int((matched["delta_r"].abs() > 1e-9).sum())),
        "original_only": int(len(original_only)),
        "veto_only": int(len(veto_only)),
        "matched_delta_r": round(float(matched["delta_r"].sum()), 4),
        "original_only_pnl_r": round(float(original_only["pnl_r"].sum()), 4),
        "veto_only_pnl_r": round(float(veto_only["pnl_r"].sum()), 4),
        "overall_delta_r": round(float(veto["pnl_r"].sum() - original["pnl_r"].sum()), 4),
        "top_positive_month": monthly_rows[0]["month"] if monthly_rows else "",
        "top_positive_month_r": monthly_rows[0]["overall_delta_r"] if monthly_rows else 0.0,
    }
    return matched_rows, unmatched_rows, monthly_rows, chain_rows, summary


def top_rows(rows: list[dict[str, Any]], *, key: str, reverse: bool, limit: int = 8) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: item[key], reverse=reverse)[:limit]


def build_report(
    *,
    summaries: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
    chain_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# 96-98 Veto vs Original 3-Bar LONG Chain Diff",
        "",
        "- 比较对象只限 `LONG`，因为这轮 `96-98 veto` 只改 `LONG` 的 post-TP1 管理。",
        "- `matched` 指完全相同的 `signal_time + entry_time + side` 共享交易；`delta_r` 反映的是同一笔交易本身被修好或修坏了多少。",
        "- `veto_only / original_only` 反映的是 sequence-aware 下，由于 `veto` 更早回到 `BE` 或更早结束持仓，释放或失去了哪些后续 `LONG` 机会。",
        "- `chain` = 共享 anchor 的直接 `delta_r` + 被 `veto` 释放的新后续交易 - 被 `veto` 反过来挡掉的旧后续交易。",
        "",
        "## Summary",
        "",
        markdown_table(
            summaries,
            [
                ("window", "Window"),
                ("original_trades", "Original 3-Bar Trades"),
                ("veto_trades", "Veto Trades"),
                ("matched_trades", "Matched"),
                ("changed_matched_trades", "Changed Matched"),
                ("original_only", "Original Only"),
                ("veto_only", "Veto Only"),
                ("matched_delta_r", "Matched Delta R"),
                ("original_only_pnl_r", "Original Only R"),
                ("veto_only_pnl_r", "Veto Only R"),
                ("overall_delta_r", "Overall Delta R"),
            ],
        ),
        "",
        "## Monthly Contribution",
        "",
        markdown_table(
            monthly_rows,
            [
                ("window", "Window"),
                ("month", "Month"),
                ("matched_delta_r", "Matched Delta R"),
                ("original_only_pnl_r", "Original Only R"),
                ("veto_only_pnl_r", "Veto Only R"),
                ("overall_delta_r", "Overall Delta R"),
            ],
        ),
        "",
        "## Changed Matched Trades",
        "",
        markdown_table(
            top_rows([row for row in matched_rows if abs(float(row["delta_r"])) > 1e-9], key="delta_r", reverse=True, limit=12),
            [
                ("window", "Window"),
                ("signal_time", "Signal"),
                ("trend_strength", "Trend"),
                ("original_exit_reason", "Original Exit"),
                ("veto_exit_reason", "Veto Exit"),
                ("original_pnl_r", "Original R"),
                ("veto_pnl_r", "Veto R"),
                ("delta_r", "Delta R"),
                ("original_decision_reason", "Original Decision"),
                ("veto_decision_reason", "Veto Decision"),
            ],
        ),
        "",
        "## Released / Lost Follow-On Trades",
        "",
        markdown_table(
            unmatched_rows,
            [
                ("window", "Window"),
                ("origin", "Origin"),
                ("signal_time", "Signal"),
                ("trend_strength", "Trend"),
                ("exit_reason", "Exit"),
                ("pnl_r", "PnL R"),
                ("blocked_by_signal_time", "Blocked By Signal"),
                ("blocked_by_exit_reason", "Blocked By Exit"),
                ("blocked_by_pnl_r", "Blocked By R"),
            ],
        ),
        "",
        "## Chain Attribution",
        "",
        markdown_table(
            sorted(chain_rows, key=lambda item: item["net_chain_delta_r"], reverse=True),
            [
                ("window", "Window"),
                ("anchor_signal_time", "Anchor Signal"),
                ("trend_strength", "Trend"),
                ("anchor_delta_r", "Anchor Delta R"),
                ("released_trade_count", "Released Count"),
                ("released_total_r", "Released R"),
                ("blocked_trade_count", "Blocked Count"),
                ("blocked_total_r", "Blocked R"),
                ("net_chain_delta_r", "Net Chain Delta R"),
                ("original_decision_reason", "Original Decision"),
                ("veto_decision_reason", "Veto Decision"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_output_dir()

    all_matched_rows: list[dict[str, Any]] = []
    all_unmatched_rows: list[dict[str, Any]] = []
    all_monthly_rows: list[dict[str, Any]] = []
    all_chain_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for window in WINDOWS:
        matched_rows, unmatched_rows, monthly_rows, chain_rows, summary = summarize_window(window)
        all_matched_rows.extend(matched_rows)
        all_unmatched_rows.extend(unmatched_rows)
        all_monthly_rows.extend(monthly_rows)
        all_chain_rows.extend(chain_rows)
        summaries.append(summary)

    write_csv(OUTPUT_DIR / "matched_pairs.csv", all_matched_rows)
    write_csv(OUTPUT_DIR / "unmatched_trades.csv", all_unmatched_rows)
    write_csv(OUTPUT_DIR / "monthly_breakdown.csv", all_monthly_rows)
    write_csv(OUTPUT_DIR / "chain_rows.csv", all_chain_rows)
    write_csv(OUTPUT_DIR / "summary.csv", summaries)

    report = build_report(
        summaries=summaries,
        monthly_rows=all_monthly_rows,
        matched_rows=all_matched_rows,
        unmatched_rows=all_unmatched_rows,
        chain_rows=all_chain_rows,
    )
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
