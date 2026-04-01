from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIFF_DIR = ROOT / "artifacts/backtests/post_tp1_extension_diff"
SOURCE_DIR = ROOT / "artifacts/backtests/post_tp1_extension_mainline"
CACHE_FILE = ROOT / "artifacts/backtests/cache/btc_usdt_usdt_1h_20190307_20260319.csv"
OUTPUT_DIR = ROOT / "artifacts/backtests/post_tp1_extension_chain_review"


@dataclass(frozen=True)
class ChainCase:
    case_id: str
    anchor_signal_time: str
    window: str
    note: str


CASES = (
    ChainCase(
        case_id="2024-11-chain",
        anchor_signal_time="2024-11-15T10:00:00+00:00",
        window="two_year",
        note="近两年窗口里最典型的阻塞链条，3-bar 把一笔 BE 单继续拿成 TP2，但卡掉了后面两笔 baseline LONG。",
    ),
    ChainCase(
        case_id="2025-05-chain",
        anchor_signal_time="2025-05-13T12:00:00+00:00",
        window="two_year",
        note="近两年窗口里最典型的正向链条，3-bar 延长持仓后卡掉了两笔 baseline LONG，但本地净值仍为正。",
    ),
)


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


def load_csv(path: Path, *, parse_dates: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates)


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched = load_csv(
        DIFF_DIR / "matched_pairs.csv",
        parse_dates=["signal_time", "entry_time", "baseline_exit_time", "extension_exit_time"],
    )
    unmatched = load_csv(
        DIFF_DIR / "unmatched_trades.csv",
        parse_dates=[
            "signal_time",
            "entry_time",
            "exit_time",
            "blocked_by_signal_time",
            "blocked_by_entry_time",
            "blocked_by_exit_time",
        ],
    )
    cache = load_csv(CACHE_FILE, parse_dates=["timestamp"])
    cache["ema_21"] = cache["close"].ewm(span=21, adjust=False).mean()
    cache["ema_55"] = cache["close"].ewm(span=55, adjust=False).mean()
    return matched, unmatched, cache


def render_chain_svg(
    *,
    case: ChainCase,
    cache: pd.DataFrame,
    anchor: pd.Series,
    blocked: pd.DataFrame,
    output_path: Path,
) -> None:
    start = min(anchor["entry_time"], blocked["entry_time"].min()) - timedelta(hours=36)
    end = max(anchor["extension_exit_time"], blocked["exit_time"].max()) + timedelta(hours=24)
    frame = cache[(cache["timestamp"] >= start) & (cache["timestamp"] <= end)].copy().reset_index(drop=True)

    width = 1460
    height = 540
    left = 74
    right = 24
    top = 52
    bottom = 60
    inner_w = width - left - right
    inner_h = height - top - bottom
    candle_w = max(inner_w / max(len(frame), 1) * 0.72, 1.2)

    prices = list(frame["low"]) + list(frame["high"]) + list(frame["ema_21"]) + list(frame["ema_55"])
    for _, trade in blocked.iterrows():
        prices.extend([float(trade["pnl_r"])])  # no-op for consistency
    for value in (
        float(anchor["baseline_pnl_r"]),
        float(anchor["extension_pnl_r"]),
    ):
        _ = value
    price_min = float(min(prices))
    price_max = float(max(prices))
    pad = (price_max - price_min) * 0.08 if price_max > price_min else 1.0
    price_min -= pad
    price_max += pad

    def x_at(index: int) -> float:
        if len(frame) <= 1:
            return left + inner_w / 2
        return left + (index / (len(frame) - 1)) * inner_w

    def y_at(price: float) -> float:
        if price_max == price_min:
            return top + inner_h / 2
        return top + inner_h - ((price - price_min) / (price_max - price_min)) * inner_h

    def idx_for(ts: datetime) -> int:
        idx = int(frame["timestamp"].searchsorted(pd.Timestamp(ts), side="left"))
        return max(0, min(idx, len(frame) - 1))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>'
        'text{font-family:Menlo,Consolas,monospace;font-size:12px;fill:#111827}'
        '.title{font-size:18px;font-weight:700}.small{font-size:11px;fill:#4b5563}'
        '.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}'
        '.bull{stroke:#0f766e;fill:#0f766e}.bear{stroke:#b91c1c;fill:#b91c1c}'
        '.ema21{stroke:#2563eb;fill:none;stroke-width:2}.ema55{stroke:#a16207;fill:none;stroke-width:2}'
        '.anchor{stroke:#7c3aed;fill:#7c3aed}.baseline{stroke:#0f766e;fill:#0f766e}.blocked{stroke:#dc2626;fill:#dc2626}'
        '.dash{stroke-dasharray:5 4;stroke-width:1.4}'
        '</style>',
        f'<text x="{left}" y="26" class="title">{case.case_id}</text>',
        f'<text x="{left}" y="44" class="small">{case.note}</text>',
        f'<line x1="{left}" y1="{top + inner_h}" x2="{left + inner_w}" y2="{top + inner_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + inner_h}" class="axis"/>',
    ]

    for idx in range(5):
        value = price_min + (price_max - price_min) * idx / 4
        y = y_at(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + inner_w}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.0f}</text>')

    ema21_points = []
    ema55_points = []
    for idx, row in frame.iterrows():
        x = x_at(idx)
        cls = "bull" if float(row["close"]) >= float(row["open"]) else "bear"
        svg.append(
            f'<line x1="{x:.1f}" y1="{y_at(float(row["high"])):.1f}" x2="{x:.1f}" y2="{y_at(float(row["low"])):.1f}" class="{cls}"/>'
        )
        body_top = y_at(max(float(row["open"]), float(row["close"])))
        body_bottom = y_at(min(float(row["open"]), float(row["close"])))
        body_height = max(abs(body_bottom - body_top), 1.2)
        rect_y = min(body_top, body_bottom)
        svg.append(
            f'<rect x="{x - candle_w / 2:.1f}" y="{rect_y:.1f}" width="{candle_w:.1f}" height="{body_height:.1f}" class="{cls}" fill-opacity="0.55"/>'
        )
        ema21_points.append(f"{x:.1f},{y_at(float(row['ema_21'])):.1f}")
        ema55_points.append(f"{x:.1f},{y_at(float(row['ema_55'])):.1f}")
    svg.append(f'<polyline class="ema21" points="{" ".join(ema21_points)}"/>')
    svg.append(f'<polyline class="ema55" points="{" ".join(ema55_points)}"/>')

    anchor_events = [
        ("Anchor signal", anchor["signal_time"], "anchor"),
        ("Baseline exit", anchor["baseline_exit_time"], "baseline"),
        ("3-bar exit", anchor["extension_exit_time"], "anchor"),
    ]
    for label, ts, cls in anchor_events:
        idx = idx_for(ts)
        x = x_at(idx)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + inner_h}" class="{cls} dash" opacity="0.85"/>')
        svg.append(f'<text x="{x + 4:.1f}" y="{top + 14}" class="small">{label}</text>')

    for i, trade in enumerate(blocked.itertuples(index=False), start=1):
        for label, ts in (
            (f"Blocked {i} signal", trade.signal_time),
            (f"Blocked {i} exit", trade.exit_time),
        ):
            idx = idx_for(ts)
            x = x_at(idx)
            svg.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + inner_h}" class="blocked dash" opacity="0.70"/>'
            )
            offset = 28 + (i - 1) * 14 if "signal" in label else inner_h - 20 - (i - 1) * 14
            y = top + offset if "signal" in label else top + offset
            svg.append(f'<text x="{x + 4:.1f}" y="{y:.1f}" class="small">{label}</text>')

    svg.append(
        f'<rect x="{width - 290}" y="{top}" width="260" height="110" fill="#ffffff" stroke="#d1d5db" rx="8"/>'
    )
    legend_y = top + 22
    svg.append(f'<line x1="{width - 272}" y1="{legend_y}" x2="{width - 246}" y2="{legend_y}" class="ema21"/>')
    svg.append(f'<text x="{width - 236}" y="{legend_y + 4}" class="small">EMA21</text>')
    legend_y += 20
    svg.append(f'<line x1="{width - 272}" y1="{legend_y}" x2="{width - 246}" y2="{legend_y}" class="ema55"/>')
    svg.append(f'<text x="{width - 236}" y="{legend_y + 4}" class="small">EMA55</text>')
    legend_y += 20
    svg.append(f'<line x1="{width - 272}" y1="{legend_y}" x2="{width - 246}" y2="{legend_y}" class="anchor dash"/>')
    svg.append(f'<text x="{width - 236}" y="{legend_y + 4}" class="small">3-bar anchor</text>')
    legend_y += 20
    svg.append(f'<line x1="{width - 272}" y1="{legend_y}" x2="{width - 246}" y2="{legend_y}" class="blocked dash"/>')
    svg.append(f'<text x="{width - 236}" y="{legend_y + 4}" class="small">Blocked baseline signal</text>')

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_report(case_rows: list[dict[str, Any]], blocked_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Post-TP1 Opportunity Chain Review",
        "",
        "- 这次不再比较全窗口统计，而是只看 `3 bars` 最有代表性的两组阻塞链条。",
        "- 目标不是判断某条规则“看起来顺眼”，而是回答：`3 bars` 为了多拿 continuation，到底值不值得放弃后续 baseline 再入场。",
        "",
        "## Chain Summary",
        "",
        markdown_table(
            case_rows,
            [
                ("case_id", "Case"),
                ("anchor_signal_time", "Anchor Signal"),
                ("anchor_delta_r", "Anchor Delta R"),
                ("blocked_trade_count", "Blocked Trades"),
                ("blocked_total_r", "Blocked Total R"),
                ("net_chain_delta_r", "Net Chain Delta R"),
                ("judgement", "Judgement"),
            ],
        ),
        "",
    ]

    for case_row in case_rows:
        case_id = case_row["case_id"]
        lines.extend(
            [
                f"## {case_id}",
                "",
                case_row["note"],
                "",
                f"- 锚点 trade：baseline `{case_row['anchor_baseline_exit_reason']} / {case_row['anchor_baseline_r']:+.4f}R`，3-bar `{case_row['anchor_extension_exit_reason']} / {case_row['anchor_extension_r']:+.4f}R`。",
                f"- 锚点单笔增益：`{case_row['anchor_delta_r']:+.4f}R`。",
                f"- 因持仓延长被卡掉的 baseline LONG：`{case_row['blocked_trade_count']} 笔 / {case_row['blocked_total_r']:+.4f}R`。",
                f"- 链条净值：`{case_row['net_chain_delta_r']:+.4f}R`。",
                f"- 判断：{case_row['judgement']}。",
                "",
                markdown_table(
                    [row for row in blocked_rows if row["case_id"] == case_id],
                    [
                        ("signal_time", "Blocked Signal"),
                        ("entry_time", "Entry"),
                        ("exit_reason", "Baseline Exit"),
                        ("pnl_r", "Baseline R"),
                        ("blocked_by_exit_time", "Blocked Until"),
                        ("blocked_by_exit_reason", "3-Bar Exit"),
                        ("blocked_by_pnl_r", "3-Bar R"),
                    ],
                ),
                "",
                f"图：[{case_id}.svg]({(OUTPUT_DIR / f'{case_id}.svg').as_posix()})",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ensure_output_dir()
    matched, unmatched, cache = load_frames()

    case_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for case in CASES:
        anchor_ts = pd.Timestamp(case.anchor_signal_time)
        anchor = matched[(matched["window"] == case.window) & (matched["signal_time"] == anchor_ts)].copy()
        if anchor.empty:
            raise ValueError(f"Anchor trade not found for {case.case_id}")
        anchor_row = anchor.iloc[0]

        blocked = unmatched[
            (unmatched["window"] == case.window)
            & (unmatched["origin"] == "baseline_only")
            & (unmatched["blocked_by_signal_time"] == anchor_ts)
        ].copy()
        blocked = blocked.sort_values("entry_time").reset_index(drop=True)

        anchor_delta = float(anchor_row["delta_r"])
        blocked_total = float(blocked["pnl_r"].sum())
        net_chain_delta = anchor_delta - blocked_total
        judgement = "局部不值得" if net_chain_delta < 0 else "局部值得"

        case_rows.append(
            {
                "case_id": case.case_id,
                "anchor_signal_time": anchor_row["signal_time"].isoformat(),
                "anchor_baseline_exit_reason": anchor_row["baseline_exit_reason"],
                "anchor_extension_exit_reason": anchor_row["extension_exit_reason"],
                "anchor_baseline_r": round(float(anchor_row["baseline_pnl_r"]), 4),
                "anchor_extension_r": round(float(anchor_row["extension_pnl_r"]), 4),
                "anchor_delta_r": round(anchor_delta, 4),
                "blocked_trade_count": int(len(blocked)),
                "blocked_total_r": round(blocked_total, 4),
                "net_chain_delta_r": round(net_chain_delta, 4),
                "judgement": judgement,
                "note": case.note,
            }
        )

        for row in blocked.itertuples(index=False):
            blocked_rows.append(
                {
                    "case_id": case.case_id,
                    "signal_time": row.signal_time.isoformat(),
                    "entry_time": row.entry_time.isoformat(),
                    "exit_time": row.exit_time.isoformat(),
                    "exit_reason": row.exit_reason,
                    "pnl_r": round(float(row.pnl_r), 4),
                    "blocked_by_signal_time": row.blocked_by_signal_time.isoformat(),
                    "blocked_by_exit_time": row.blocked_by_exit_time.isoformat(),
                    "blocked_by_exit_reason": row.blocked_by_exit_reason,
                    "blocked_by_pnl_r": round(float(row.blocked_by_pnl_r), 4),
                }
            )

        render_chain_svg(
            case=case,
            cache=cache,
            anchor=anchor_row,
            blocked=blocked,
            output_path=OUTPUT_DIR / f"{case.case_id}.svg",
        )

    write_csv(OUTPUT_DIR / "case_summary.csv", case_rows)
    write_csv(OUTPUT_DIR / "blocked_trades.csv", blocked_rows)
    report = build_report(case_rows, blocked_rows)
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved report: {report_path}")
    print(f"Saved case summary CSV: {OUTPUT_DIR / 'case_summary.csv'}")
    print(f"Saved blocked trades CSV: {OUTPUT_DIR / 'blocked_trades.csv'}")


if __name__ == "__main__":
    main()
