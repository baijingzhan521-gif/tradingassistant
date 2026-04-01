from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SOURCE_DIR = ROOT / "artifacts/backtests/entry_attribution_matrix_2020"
OUTPUT_DIR = ROOT / "artifacts/backtests/mainline_vs_no_reversal_no_aux_2020_diagnostic"

MAINLINE = "entry_attr_r1_rf1_hs1_aux1"
NO_REV_NO_AUX = "entry_attr_r0_rf1_hs1_aux0"

PROFILE_LABELS = {
    MAINLINE: "Mainline (R1 RF1 HS1 AUX1)",
    NO_REV_NO_AUX: "No-Reversal-No-Aux (R0 RF1 HS1 AUX0)",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trades() -> pd.DataFrame:
    trades = pd.read_csv(
        SOURCE_DIR / "entry_attribution_oos_trades.csv",
        parse_dates=["signal_time", "entry_time", "exit_time"],
    )
    trades = trades[trades["strategy_profile"].isin([MAINLINE, NO_REV_NO_AUX])].copy()
    trades["year"] = trades["entry_time"].dt.year
    trades["quarter"] = trades["entry_time"].dt.to_period("Q").astype(str)
    trades["profile_label"] = trades["strategy_profile"].map(PROFILE_LABELS)
    return trades


def load_folds() -> pd.DataFrame:
    folds = pd.read_csv(SOURCE_DIR / "entry_attribution_folds.csv")
    folds = folds[folds["profile"].isin([MAINLINE, NO_REV_NO_AUX])].copy()
    folds["profile_label"] = folds["profile"].map(PROFILE_LABELS)
    return folds


def nice_range(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        pad = abs(lo) * 0.1 if lo else 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.1
    return lo - pad, hi + pad


def svg_line_chart(
    *,
    series: dict[str, list[tuple[str, float]]],
    title: str,
    y_label: str,
    path: Path,
) -> None:
    width = 980
    height = 420
    left = 72
    right = 24
    top = 44
    bottom = 58
    inner_w = width - left - right
    inner_h = height - top - bottom
    colors = ["#0f766e", "#b45309", "#1d4ed8", "#7c2d12"]
    all_values = [point[1] for points in series.values() for point in points]
    y_min, y_max = nice_range(all_values or [0.0, 1.0])

    def y_to_px(value: float) -> float:
        if y_max == y_min:
            return top + inner_h / 2
        return top + inner_h - ((value - y_min) / (y_max - y_min)) * inner_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Menlo,Consolas,monospace;font-size:12px;fill:#111827}.title{font-size:18px;font-weight:700}.small{font-size:11px;fill:#4b5563}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.legend{font-size:12px}</style>',
        f'<text x="{left}" y="24" class="title">{title}</text>',
        f'<text x="18" y="{top + inner_h / 2}" transform="rotate(-90 18 {top + inner_h / 2})" class="small">{y_label}</text>',
        f'<line x1="{left}" y1="{top + inner_h}" x2="{left + inner_w}" y2="{top + inner_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + inner_h}" class="axis"/>',
    ]

    for idx in range(5):
        value = y_min + (y_max - y_min) * idx / 4
        y = y_to_px(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + inner_w}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.1f}</text>')

    for series_idx, (label, points) in enumerate(series.items()):
        color = colors[series_idx % len(colors)]
        coords = []
        n = max(len(points), 2)
        for idx, (_, value) in enumerate(points):
            x = left + (idx / (n - 1)) * inner_w
            y = y_to_px(value)
            coords.append(f"{x:.1f},{y:.1f}")
        if coords:
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(coords)}"/>')
            last_x, last_y = coords[-1].split(",")
            svg.append(f'<circle cx="{last_x}" cy="{last_y}" r="3.5" fill="{color}"/>')
            svg.append(f'<text x="{float(last_x) + 8:.1f}" y="{float(last_y) + 4:.1f}" class="legend">{label}</text>')

    first_series = next(iter(series.values()), [])
    if first_series:
        tick_positions = []
        if len(first_series) <= 8:
            tick_positions = list(range(len(first_series)))
        else:
            tick_positions = sorted(set([0, len(first_series) // 4, len(first_series) // 2, (3 * len(first_series)) // 4, len(first_series) - 1]))
        for idx in tick_positions:
            x = left + (idx / max(len(first_series) - 1, 1)) * inner_w
            label = first_series[idx][0]
            svg.append(f'<text x="{x:.1f}" y="{top + inner_h + 22}" text-anchor="middle" class="small">{label}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def svg_grouped_bar_chart(
    *,
    rows: list[dict[str, object]],
    categories: list[str],
    series_keys: list[tuple[str, str]],
    title: str,
    y_label: str,
    path: Path,
) -> None:
    width = 980
    height = 420
    left = 72
    right = 24
    top = 44
    bottom = 58
    inner_w = width - left - right
    inner_h = height - top - bottom
    colors = ["#0f766e", "#b45309", "#1d4ed8", "#7c2d12"]
    values = [float(row[key]) for row in rows for key, _ in series_keys]
    y_min, y_max = nice_range(values or [0.0, 1.0])
    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)

    def y_to_px(value: float) -> float:
        if y_max == y_min:
            return top + inner_h / 2
        return top + inner_h - ((value - y_min) / (y_max - y_min)) * inner_h

    zero_y = y_to_px(0.0)
    group_w = inner_w / max(len(categories), 1)
    bar_w = group_w / (len(series_keys) + 1)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Menlo,Consolas,monospace;font-size:12px;fill:#111827}.title{font-size:18px;font-weight:700}.small{font-size:11px;fill:#4b5563}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.legend{font-size:12px}</style>',
        f'<text x="{left}" y="24" class="title">{title}</text>',
        f'<text x="18" y="{top + inner_h / 2}" transform="rotate(-90 18 {top + inner_h / 2})" class="small">{y_label}</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + inner_w}" y2="{zero_y:.1f}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + inner_h}" class="axis"/>',
    ]

    for idx in range(5):
        value = y_min + (y_max - y_min) * idx / 4
        y = y_to_px(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + inner_w}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.1f}</text>')

    for cat_idx, category in enumerate(categories):
        row = rows[cat_idx]
        group_start = left + cat_idx * group_w
        svg.append(f'<text x="{group_start + group_w / 2:.1f}" y="{top + inner_h + 22}" text-anchor="middle" class="small">{category}</text>')
        for series_idx, (key, legend) in enumerate(series_keys):
            color = colors[series_idx % len(colors)]
            value = float(row[key])
            x = group_start + bar_w * (series_idx + 0.5)
            y = y_to_px(max(value, 0.0))
            h = abs(y_to_px(value) - zero_y)
            rect_y = min(y, zero_y)
            svg.append(f'<rect x="{x:.1f}" y="{rect_y:.1f}" width="{bar_w * 0.8:.1f}" height="{max(h, 1):.1f}" fill="{color}"/>')

    for series_idx, (_, legend) in enumerate(series_keys):
        color = colors[series_idx % len(colors)]
        legend_x = left + series_idx * 180
        svg.append(f'<rect x="{legend_x}" y="{height - 24}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 20}" y="{height - 12}" class="legend">{legend}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_cumulative_rows(trades: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for profile in [NO_REV_NO_AUX, MAINLINE]:
        sub = trades[trades["strategy_profile"] == profile].sort_values("exit_time").copy()
        sub["cum_r"] = sub["pnl_r"].cumsum()
        for idx, row in enumerate(sub.itertuples(index=False), start=1):
            rows.append(
                {
                    "profile": profile,
                    "profile_label": PROFILE_LABELS[profile],
                    "trade_index": idx,
                    "exit_time": row.exit_time.isoformat(),
                    "cum_r": round(float(row.cum_r), 4),
                }
            )
    return rows


def build_year_rows(trades: pd.DataFrame) -> list[dict[str, object]]:
    pivot = (
        trades.groupby(["year", "strategy_profile"])
        .agg(trades=("pnl_r", "size"), cum_r=("pnl_r", "sum"), avg_r=("pnl_r", "mean"))
        .reset_index()
        .pivot(index="year", columns="strategy_profile", values="cum_r")
        .fillna(0.0)
        .reset_index()
    )
    rows = []
    for row in pivot.itertuples(index=False):
        rows.append(
            {
                "year": int(row.year),
                "no_rev_no_aux_cum_r": round(float(getattr(row, NO_REV_NO_AUX)), 4),
                "mainline_cum_r": round(float(getattr(row, MAINLINE)), 4),
                "diff": round(float(getattr(row, NO_REV_NO_AUX) - getattr(row, MAINLINE)), 4),
            }
        )
    return rows


def build_year_side_rows(trades: pd.DataFrame) -> list[dict[str, object]]:
    grouped = (
        trades.groupby(["year", "side", "strategy_profile"])
        .agg(trades=("pnl_r", "size"), cum_r=("pnl_r", "sum"))
        .reset_index()
    )
    pivot = grouped.pivot_table(index=["year", "side"], columns="strategy_profile", values="cum_r", fill_value=0.0).reset_index()
    rows = []
    for row in pivot.itertuples(index=False):
        rows.append(
            {
                "year": int(row.year),
                "side": row.side,
                "no_rev_no_aux_cum_r": round(float(getattr(row, NO_REV_NO_AUX)), 4),
                "mainline_cum_r": round(float(getattr(row, MAINLINE)), 4),
                "diff": round(float(getattr(row, NO_REV_NO_AUX) - getattr(row, MAINLINE)), 4),
            }
        )
    return rows


def build_fold_rows(folds: pd.DataFrame) -> list[dict[str, object]]:
    pivot = folds.pivot(index="fold", columns="profile", values=["test_cum_r", "test_pf", "test_exp_r", "test_dd_r"])
    pivot.columns = [f"{metric}_{profile}" for metric, profile in pivot.columns]
    pivot = pivot.reset_index()
    rows = []
    for row in pivot.itertuples(index=False):
        no_cum = float(getattr(row, f"test_cum_r_{NO_REV_NO_AUX}"))
        main_cum = float(getattr(row, f"test_cum_r_{MAINLINE}"))
        rows.append(
            {
                "fold": int(row.fold),
                "no_rev_no_aux_cum_r": round(no_cum, 4),
                "mainline_cum_r": round(main_cum, 4),
                "cum_r_diff": round(no_cum - main_cum, 4),
                "no_rev_no_aux_pf": round(float(getattr(row, f"test_pf_{NO_REV_NO_AUX}")), 4),
                "mainline_pf": round(float(getattr(row, f"test_pf_{MAINLINE}")), 4),
            }
        )
    return rows


def build_side_distribution_rows(trades: pd.DataFrame) -> list[dict[str, object]]:
    grouped = trades.groupby(["strategy_profile", "side"]).agg(
        trades=("pnl_r", "size"),
        cum_r=("pnl_r", "sum"),
        avg_r=("pnl_r", "mean"),
        median_r=("pnl_r", "median"),
        win_rate=("pnl_r", lambda s: float((s > 0).mean())),
        avg_hold=("bars_held", "mean"),
    )
    rows = []
    for (profile, side), values in grouped.iterrows():
        rows.append(
            {
                "profile_label": PROFILE_LABELS[profile],
                "side": side,
                "trades": int(values["trades"]),
                "cum_r": round(float(values["cum_r"]), 4),
                "avg_r": round(float(values["avg_r"]), 4),
                "median_r": round(float(values["median_r"]), 4),
                "win_rate": round(float(values["win_rate"]), 4),
                "avg_hold_bars": round(float(values["avg_hold"]), 2),
            }
        )
    return rows


def build_exit_rows(trades: pd.DataFrame) -> list[dict[str, object]]:
    grouped = trades.groupby(["strategy_profile", "exit_reason"]).agg(
        trades=("pnl_r", "size"),
        cum_r=("pnl_r", "sum"),
        avg_r=("pnl_r", "mean"),
    )
    rows = []
    for (profile, reason), values in grouped.iterrows():
        rows.append(
            {
                "profile_label": PROFILE_LABELS[profile],
                "exit_reason": reason,
                "trades": int(values["trades"]),
                "cum_r": round(float(values["cum_r"]), 4),
                "avg_r": round(float(values["avg_r"]), 4),
            }
        )
    return rows


def render_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    ensure_output_dir()
    trades = load_trades()
    folds = load_folds()

    cumulative_rows = build_cumulative_rows(trades)
    year_rows = build_year_rows(trades)
    year_side_rows = build_year_side_rows(trades)
    fold_rows = build_fold_rows(folds)
    side_rows = build_side_distribution_rows(trades)
    exit_rows = build_exit_rows(trades)

    write_csv(OUTPUT_DIR / "cumulative_oos.csv", cumulative_rows)
    write_csv(OUTPUT_DIR / "yearly_summary.csv", year_rows)
    write_csv(OUTPUT_DIR / "yearly_side_summary.csv", year_side_rows)
    write_csv(OUTPUT_DIR / "fold_diff.csv", fold_rows)
    write_csv(OUTPUT_DIR / "side_distribution.csv", side_rows)
    write_csv(OUTPUT_DIR / "exit_distribution.csv", exit_rows)

    cumulative_series = {}
    for profile in [NO_REV_NO_AUX, MAINLINE]:
        sub = [row for row in cumulative_rows if row["profile"] == profile]
        cumulative_series[PROFILE_LABELS[profile]] = [(str(row["trade_index"]), float(row["cum_r"])) for row in sub]
    svg_line_chart(
        series=cumulative_series,
        title="OOS Cumulative R Curve",
        y_label="Cumulative R",
        path=OUTPUT_DIR / "cumulative_oos.svg",
    )

    svg_grouped_bar_chart(
        rows=year_rows,
        categories=[str(row["year"]) for row in year_rows],
        series_keys=[
            ("no_rev_no_aux_cum_r", PROFILE_LABELS[NO_REV_NO_AUX]),
            ("mainline_cum_r", PROFILE_LABELS[MAINLINE]),
        ],
        title="Yearly OOS Cumulative R",
        y_label="Cumulative R",
        path=OUTPUT_DIR / "yearly_oos.svg",
    )

    svg_grouped_bar_chart(
        rows=fold_rows,
        categories=[f"F{row['fold']}" for row in fold_rows],
        series_keys=[
            ("cum_r_diff", "No-Reversal-No-Aux minus Mainline"),
        ],
        title="Fold-Level OOS Difference",
        y_label="Delta Cumulative R",
        path=OUTPUT_DIR / "fold_diff.svg",
    )

    best_positive_folds = [row for row in fold_rows if float(row["cum_r_diff"]) > 0]
    worst_negative_folds = [row for row in fold_rows if float(row["cum_r_diff"]) < 0]
    best_positive_folds = sorted(best_positive_folds, key=lambda row: float(row["cum_r_diff"]), reverse=True)[:5]
    worst_negative_folds = sorted(worst_negative_folds, key=lambda row: float(row["cum_r_diff"]))[:5]

    report = "\n".join(
        [
            "# Mainline vs No-Reversal-No-Aux",
            "",
            "## 这轮在做什么",
            "",
            "- `mainline` 指当前主线：`R1 RF1 HS1 AUX1`，也就是 `reversal=on / regained_fast=on / held_slow=on / auxiliary=on`。",
            "- `no-reversal-no-aux` 指新候选：`R0 RF1 HS1 AUX0`，也就是 `reversal=off / regained_fast=on / held_slow=on / auxiliary=off`。",
            "- 这里的 `R / RF / HS / AUX` 只是 entry 组件开关，不是新的策略家族：",
            "- `R` = `reversal`，是否要求 reversal candle。",
            "- `RF` = `regained_fast`，是否要求收回/跌回快均线 `EMA21`。",
            "- `HS` = `held_slow`，是否要求站稳/失守慢均线 `EMA55` 或局部结构修复。",
            "- `AUX` = `auxiliary`，是否要求“没创新低/高 + rejection + volume contraction”这类辅助确认。",
            "",
            "## 核心判断",
            "",
            "- `no-reversal-no-aux` 在 `2020-03-19 -> 2026-03-19` 的 OOS 总体是 `+14.2360R / PF 1.1221 / DD 12.1964R`。",
            "- `mainline` 在同口径 OOS 总体是 `-1.1450R / PF 0.9898 / DD 25.6158R`。",
            "- 差异不是均匀发生的。新候选真正“救回主线”的阶段主要集中在 `2022` 的空头段，以及部分 `2026` 初的空头段。",
            "- `2024` 反而是 mainline 更强，说明这不是单向碾压，而是风格迁移：新候选更擅长中长期下跌/反弹反复阶段，mainline 更擅长近两年的高质量顺势多头窗口。",
            "",
            "## 图表",
            "",
            f"![Cumulative OOS]({(OUTPUT_DIR / 'cumulative_oos.svg').as_posix()})",
            "",
            f"![Yearly OOS]({(OUTPUT_DIR / 'yearly_oos.svg').as_posix()})",
            "",
            f"![Fold Diff]({(OUTPUT_DIR / 'fold_diff.svg').as_posix()})",
            "",
            "## 年度阶段对比",
            "",
            render_table(
                year_rows,
                [
                    ("year", "年份"),
                    ("no_rev_no_aux_cum_r", "No-Rev-No-Aux"),
                    ("mainline_cum_r", "Mainline"),
                    ("diff", "差值"),
                ],
            ),
            "",
            "## Fold 差异最大的阶段",
            "",
            "正向差异最大的 5 个 fold：",
            render_table(
                best_positive_folds,
                [
                    ("fold", "Fold"),
                    ("no_rev_no_aux_cum_r", "No-Rev-No-Aux"),
                    ("mainline_cum_r", "Mainline"),
                    ("cum_r_diff", "差值"),
                    ("no_rev_no_aux_pf", "新候选PF"),
                    ("mainline_pf", "主线PF"),
                ],
            ),
            "",
            "负向差异最大的 5 个 fold：",
            render_table(
                worst_negative_folds,
                [
                    ("fold", "Fold"),
                    ("no_rev_no_aux_cum_r", "No-Rev-No-Aux"),
                    ("mainline_cum_r", "Mainline"),
                    ("cum_r_diff", "差值"),
                    ("no_rev_no_aux_pf", "新候选PF"),
                    ("mainline_pf", "主线PF"),
                ],
            ),
            "",
            "## 按方向的交易分布",
            "",
            render_table(
                side_rows,
                [
                    ("profile_label", "Profile"),
                    ("side", "方向"),
                    ("trades", "交易数"),
                    ("cum_r", "累计R"),
                    ("avg_r", "平均R"),
                    ("median_r", "中位R"),
                    ("win_rate", "胜率"),
                    ("avg_hold_bars", "平均持有bar"),
                ],
            ),
            "",
            "## 按退出原因的分布",
            "",
            render_table(
                exit_rows,
                [
                    ("profile_label", "Profile"),
                    ("exit_reason", "退出原因"),
                    ("trades", "交易数"),
                    ("cum_r", "累计R"),
                    ("avg_r", "平均R"),
                ],
            ),
            "",
            "## 解释",
            "",
            "- 新候选相对 mainline 的主要增益来自 `SHORT` 侧：`+14.08R` 对 `-5.52R`。",
            "- `LONG` 侧其实不是新候选更强，mainline 的 `LONG` 仍然更好：`+4.38R` 对 `+0.16R`。",
            "- 所以新候选不是“全面更优”，而是“用牺牲一部分多头质量，换回大量空头阶段失真”。",
            "- 从退出分布看，两者的 `stop_loss` 平均单笔损失几乎一样，差别主要来自：新候选拿到了更多 `SHORT 1.5R` 和 `breakeven_after_tp1` 的有效兑现，而不是单笔止损更小。",
            "",
            "## 原始文件",
            "",
            "- `cumulative_oos.csv`",
            "- `yearly_summary.csv`",
            "- `yearly_side_summary.csv`",
            "- `fold_diff.csv`",
            "- `side_distribution.csv`",
            "- `exit_distribution.csv`",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
