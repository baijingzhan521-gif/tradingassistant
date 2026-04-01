from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging
from app.services.derivatives_predictive_power_service import (
    DerivativesPredictivePowerService,
)


INPUT_PATH = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_hourly_state" / "research_table.csv"
OUTPUT_DIR = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_predictive_power"


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


def build_report(
    availability_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    highlighted_state_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# BTC Derivatives Predictive Power Study",
            "",
            "- 这轮不是策略回测，而是先回答单标的小时面板里的 derivatives state 对未来 `1h / 4h / 24h` 收益与波动是否有条件预测力。",
            "- 数据源仍是 Bybit `BTCUSDT linear` 小时面板。",
            "- `funding` 用滚动 `7d z-score`；`OI` 先看 `1h` 变化本身，再用 `7d z-score` 做状态分桶；`mark-index premium` 和 `basis` 都按 `7d z-score` 做状态分桶。",
            "- `liquidation burst` 当前没有纳入历史回测面板，因为官方只提供实时 websocket 流，缺少同口径长历史回补接口。",
            "- `low / mid / high` 状态定义不是全样本分位数，而是固定阈值：`z <= -1`、`-1 < z < 1`、`z >= 1`。",
            "",
            "## Availability",
            "",
            markdown_table(
                availability_rows,
                [
                    ("feature", "Feature"),
                    ("available", "Available"),
                    ("rows", "Rows"),
                    ("note", "Note"),
                ],
            ),
            "",
            "## Linear Correlation",
            "",
            markdown_table(
                correlation_rows,
                [
                    ("feature", "Feature"),
                    ("horizon_hours", "Horizon h"),
                    ("observations", "Obs"),
                    ("corr_forward_bps", "Corr Fwd"),
                    ("corr_abs_forward_bps", "Corr |Fwd|"),
                ],
            ),
            "",
            "## State Edge Summary",
            "",
            markdown_table(
                edge_rows,
                [
                    ("feature", "Feature"),
                    ("horizon_hours", "Horizon h"),
                    ("high_minus_low_forward_bps", "High-Low Fwd bps"),
                    ("high_minus_mid_forward_bps", "High-Mid Fwd bps"),
                    ("low_minus_mid_forward_bps", "Low-Mid Fwd bps"),
                    ("extreme_to_mid_abs_vol_ratio", "Extreme/Mid |Vol|"),
                ],
            ),
            "",
            "## Highlighted States",
            "",
            markdown_table(
                highlighted_state_rows,
                [
                    ("feature", "Feature"),
                    ("horizon_hours", "Horizon h"),
                    ("state", "State"),
                    ("observations", "Obs"),
                    ("feature_mean", "Feature Mean"),
                    ("mean_forward_bps", "Mean Fwd bps"),
                    ("up_rate_pct", "Up Rate %"),
                    ("mean_abs_forward_bps", "Mean |Fwd| bps"),
                ],
            ),
            "",
        ]
    )


def main() -> None:
    configure_logging()
    ensure_output_dir()

    frame = pd.read_csv(INPUT_PATH)
    service = DerivativesPredictivePowerService()

    availability_rows = service.summarize_availability(frame)
    correlation_rows = service.summarize_correlations(frame)
    state_rows = service.summarize_state_tables(frame)
    edge_rows = service.summarize_feature_edges(state_rows)

    state_frame = pd.DataFrame(state_rows)
    highlighted_state_rows: list[dict[str, Any]] = []
    if not state_frame.empty:
        state_frame = state_frame[state_frame["observations"] >= 200]
        strongest_directional = state_frame.reindex(state_frame["mean_forward_bps"].abs().sort_values(ascending=False).index).head(10)
        strongest_vol = state_frame.sort_values("mean_abs_forward_bps", ascending=False).head(10)
        highlighted_state_rows = (
            pd.concat([strongest_directional, strongest_vol], ignore_index=True)
            .drop_duplicates(subset=["feature", "horizon_hours", "state"])
            .to_dict("records")
        )

    write_csv(OUTPUT_DIR / "availability.csv", availability_rows)
    write_csv(OUTPUT_DIR / "correlation_summary.csv", correlation_rows)
    write_csv(OUTPUT_DIR / "state_summary.csv", state_rows)
    write_csv(OUTPUT_DIR / "edge_summary.csv", edge_rows)
    report = build_report(availability_rows, correlation_rows, edge_rows, highlighted_state_rows)
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
