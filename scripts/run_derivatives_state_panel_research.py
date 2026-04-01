from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import SessionLocal, init_db
from app.core.logging import configure_logging
from app.data.bybit_derivatives_client import BybitDerivativesClient
from app.services.derivatives_state_service import DerivativesStateService


OUTPUT_DIR = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_hourly_state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Bybit BTCUSDT derivatives state panel and research table.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2024-01-01T00:00:00+00:00")
    parser.add_argument("--end", default=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat())
    parser.add_argument("--skip-db", action="store_true", help="Do not persist the hourly panel into SQLite.")
    return parser.parse_args()


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
    body: list[str] = [header, divider]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(body)


def summarize_panel(panel: pd.DataFrame, research: pd.DataFrame, *, symbol: str, start: str, end: str) -> dict[str, Any]:
    missing_core_pct = (
        panel[["mark_close", "index_close", "open_interest"]].isna().any(axis=1).mean() * 100.0 if not panel.empty else 0.0
    )
    usable_rows = research[
        [
            "funding_rate_z_7d",
            "open_interest_change_z_7d",
            "basis_proxy_bps_z_7d",
            "forward_index_return_bps_24h",
        ]
    ].dropna()
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "hours": int(len(panel)),
        "usable_rows_24h": int(len(usable_rows)),
        "funding_events": int(panel["funding_rate_event"].notna().sum()),
        "missing_core_pct": round(float(missing_core_pct), 4),
        "avg_funding_bps": round(float(panel["funding_rate"].dropna().mean() * 10000.0), 4),
        "avg_basis_proxy_bps": round(float(panel["basis_proxy_bps"].dropna().mean()), 4),
        "avg_oi_change_1h_pct": round(float(panel["open_interest_change_1h_pct"].dropna().mean()), 4),
        "avg_abs_forward_24h_bps": round(float(research["forward_abs_index_return_bps_24h"].dropna().mean()), 4),
    }


def bucket_feature(
    research: pd.DataFrame,
    *,
    feature: str,
    label: str,
    quantiles: int = 5,
) -> list[dict[str, Any]]:
    subset = research[
        [
            "timestamp",
            feature,
            "forward_index_return_bps_1h",
            "forward_index_return_bps_4h",
            "forward_index_return_bps_24h",
            "forward_abs_index_return_bps_24h",
        ]
    ].dropna()
    if len(subset) < quantiles * 20:
        return []

    ranked = subset[feature].rank(method="first")
    subset["bucket"] = pd.qcut(ranked, q=quantiles, labels=False) + 1

    rows: list[dict[str, Any]] = []
    for bucket, group in subset.groupby("bucket", sort=True):
        rows.append(
            {
                "feature": label,
                "bucket": int(bucket),
                "observations": int(len(group)),
                "feature_mean": round(float(group[feature].mean()), 4),
                "forward_1h_bps": round(float(group["forward_index_return_bps_1h"].mean()), 4),
                "forward_4h_bps": round(float(group["forward_index_return_bps_4h"].mean()), 4),
                "forward_24h_bps": round(float(group["forward_index_return_bps_24h"].mean()), 4),
                "forward_abs_24h_bps": round(float(group["forward_abs_index_return_bps_24h"].mean()), 4),
            }
        )
    return rows


def build_report(summary: dict[str, Any], bucket_rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Bybit BTCUSDT Derivatives State Panel",
            "",
            "- 这不是交易策略回测，而是第一阶段的数据层与研究表构建。",
            "- 当前只接入 Bybit `BTCUSDT linear` 历史接口，用于研究 `funding / OI / mark-index premium` 状态变量。",
            "- 这版还没有历史 liquidation，因为公开 REST 历史回补不完整；不要把这份面板误解成“全量衍生品状态”。",
            "- 这版 label 直接用 Bybit `index_close` 的未来收益，不是 Binance 主线的实盘部署回测口径。",
            "",
            "## Coverage",
            "",
            markdown_table(
                [summary],
                [
                    ("symbol", "Symbol"),
                    ("start", "Start"),
                    ("end", "End"),
                    ("hours", "Hours"),
                    ("usable_rows_24h", "Usable Rows 24h"),
                    ("funding_events", "Funding Events"),
                    ("missing_core_pct", "Missing Core %"),
                    ("avg_funding_bps", "Avg Funding bps"),
                    ("avg_basis_proxy_bps", "Avg Basis bps"),
                    ("avg_oi_change_1h_pct", "Avg OI Δ1h %"),
                    ("avg_abs_forward_24h_bps", "Avg |Fwd 24h| bps"),
                ],
            ),
            "",
            "## Feature Buckets",
            "",
            markdown_table(
                bucket_rows,
                [
                    ("feature", "Feature"),
                    ("bucket", "Bucket"),
                    ("observations", "Obs"),
                    ("feature_mean", "Feature Mean"),
                    ("forward_1h_bps", "Fwd 1h bps"),
                    ("forward_4h_bps", "Fwd 4h bps"),
                    ("forward_24h_bps", "Fwd 24h bps"),
                    ("forward_abs_24h_bps", "|Fwd 24h| bps"),
                ],
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    ensure_output_dir()
    init_db()

    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))

    client = BybitDerivativesClient()
    service = DerivativesStateService(client)
    try:
        panel = service.build_hourly_panel(symbol=args.symbol, start=start, end=end)
        research = service.build_research_table(panel)

        if not args.skip_db:
            with SessionLocal() as db:
                service.persist_hourly_panel(db, venue="bybit", symbol=args.symbol, interval="1h", panel=panel)

        panel.to_csv(OUTPUT_DIR / "panel.csv", index=False)
        research.to_csv(OUTPUT_DIR / "research_table.csv", index=False)

        summary = summarize_panel(panel, research, symbol=args.symbol, start=args.start, end=args.end)
        bucket_rows: list[dict[str, Any]] = []
        bucket_rows.extend(bucket_feature(research, feature="funding_rate_z_7d", label="funding_z_7d"))
        bucket_rows.extend(bucket_feature(research, feature="open_interest_change_z_7d", label="oi_change_z_7d"))
        bucket_rows.extend(bucket_feature(research, feature="basis_proxy_bps_z_7d", label="basis_z_7d"))
        bucket_rows.extend(bucket_feature(research, feature="mark_index_spread_bps_z_7d", label="mark_spread_z_7d"))

        write_csv(OUTPUT_DIR / "bucket_summary.csv", bucket_rows)
        (OUTPUT_DIR / "report.md").write_text(build_report(summary, bucket_rows), encoding="utf-8")
    finally:
        client.close()


if __name__ == "__main__":
    main()
