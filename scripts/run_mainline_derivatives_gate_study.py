from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestReport, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


SYMBOL = "BTC/USDT:USDT"
PROFILE = "swing_trend_long_regime_gate_v1"
OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "mainline_derivatives_gate_study"
DERIVATIVES_RESEARCH_PATH = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_hourly_state" / "research_table.csv"
WINDOWS = {
    "two_year": (
        datetime(2024, 3, 19, tzinfo=timezone.utc),
        datetime(2026, 3, 19, tzinfo=timezone.utc),
    ),
    "full_2020": (
        datetime(2020, 3, 19, tzinfo=timezone.utc),
        datetime(2026, 3, 19, tzinfo=timezone.utc),
    ),
}
FEATURES = [
    "funding_rate_z_7d",
    "basis_proxy_bps_z_7d",
    "mark_index_spread_bps_z_7d",
    "open_interest_change_z_7d",
    "open_interest_notional_change_z_7d",
]


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
        formatted: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def make_backtest_service() -> BacktestService:
    assumptions = BacktestAssumptions(
        exit_profile="long_scaled1_3_short_fixed1_5",
        take_profit_mode="scaled",
        long_exit={"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        short_exit={"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
        swing_detection_mode="confirmed",
        cache_dir="artifacts/backtests/cache",
    )
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )


def run_mainline(window: str) -> BacktestReport:
    start, end = WINDOWS[window]
    return make_backtest_service().run(
        exchange="binance",
        market_type="perpetual",
        symbols=[SYMBOL],
        strategy_profiles=[PROFILE],
        start=start,
        end=end,
    )


def trades_frame(report: BacktestReport) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(item) for item in report.trades])
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame["signal_month"] = frame["signal_time"].dt.strftime("%Y-%m")
    frame["win"] = frame["pnl_r"] > 0
    return frame


def load_research_table() -> pd.DataFrame:
    frame = pd.read_csv(DERIVATIVES_RESEARCH_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_quantile_edges(research: pd.DataFrame) -> dict[str, list[float]]:
    edges: dict[str, list[float]] = {}
    for feature in FEATURES:
        values = research[feature].dropna()
        edges[feature] = values.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    return edges


def assign_bucket(series: pd.Series, edges: list[float]) -> pd.Series:
    if len(edges) != 4:
        raise ValueError("Need 4 edges for quintile assignment")
    return pd.cut(
        series,
        bins=[float("-inf"), edges[0], edges[1], edges[2], edges[3], float("inf")],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype("float")


def join_trades_with_derivatives(
    *,
    window: str,
    trades: pd.DataFrame,
    research: pd.DataFrame,
    quantile_edges: dict[str, list[float]],
) -> pd.DataFrame:
    join_columns = ["timestamp"] + FEATURES
    enriched = trades.merge(research[join_columns], left_on="signal_time", right_on="timestamp", how="left")
    enriched["window"] = window
    for feature in FEATURES:
        enriched[f"{feature}_bucket"] = assign_bucket(enriched[feature], quantile_edges[feature])
    return enriched


def summarize_subset(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "expectancy_r": 0.0,
            "cum_r": 0.0,
            "avg_r": 0.0,
        }
    return {
        "trades": int(len(frame)),
        "win_rate_pct": round(float((frame["pnl_r"] > 0).mean() * 100.0), 2),
        "expectancy_r": round(float(frame["pnl_r"].mean()), 4),
        "cum_r": round(float(frame["pnl_r"].sum()), 4),
        "avg_r": round(float(frame["pnl_r"].mean()), 4),
    }


def build_bucket_rows(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window, window_frame in enriched.groupby("window"):
        for side_label, side_frame in [("ALL", window_frame), ("LONG", window_frame[window_frame["side"] == "LONG"]), ("SHORT", window_frame[window_frame["side"] == "SHORT"])]:
            for feature in FEATURES:
                feature_frame = side_frame.dropna(subset=[feature, f"{feature}_bucket"])
                for bucket in range(1, 6):
                    bucket_frame = feature_frame[feature_frame[f"{feature}_bucket"] == float(bucket)]
                    if bucket_frame.empty:
                        continue
                    summary = summarize_subset(bucket_frame)
                    rows.append(
                        {
                            "window": window,
                            "side": side_label,
                            "feature": feature,
                            "bucket": bucket,
                            "feature_mean": round(float(bucket_frame[feature].mean()), 4),
                            **summary,
                        }
                    )
    return rows


def build_candidate_rows(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = [
        (
            "veto_long_funding_q5",
            lambda row: row["side"] == "LONG" and row["funding_rate_z_7d_bucket"] == 5.0,
        ),
        (
            "veto_long_basis_q5",
            lambda row: row["side"] == "LONG" and row["basis_proxy_bps_z_7d_bucket"] == 5.0,
        ),
        (
            "veto_long_mark_spread_q5",
            lambda row: row["side"] == "LONG" and row["mark_index_spread_bps_z_7d_bucket"] == 5.0,
        ),
        (
            "veto_long_crowded_q5q5",
            lambda row: row["side"] == "LONG"
            and row["funding_rate_z_7d_bucket"] == 5.0
            and row["basis_proxy_bps_z_7d_bucket"] == 5.0,
        ),
        (
            "veto_short_funding_q1",
            lambda row: row["side"] == "SHORT" and row["funding_rate_z_7d_bucket"] == 1.0,
        ),
        (
            "veto_short_basis_q1",
            lambda row: row["side"] == "SHORT" and row["basis_proxy_bps_z_7d_bucket"] == 1.0,
        ),
        (
            "veto_short_mark_spread_q1",
            lambda row: row["side"] == "SHORT" and row["mark_index_spread_bps_z_7d_bucket"] == 1.0,
        ),
        (
            "veto_short_crowded_q1q1",
            lambda row: row["side"] == "SHORT"
            and row["funding_rate_z_7d_bucket"] == 1.0
            and row["basis_proxy_bps_z_7d_bucket"] == 1.0,
        ),
        (
            "veto_long_oi_down_q1",
            lambda row: row["side"] == "LONG" and row["open_interest_change_z_7d_bucket"] == 1.0,
        ),
        (
            "veto_short_oi_up_q5",
            lambda row: row["side"] == "SHORT" and row["open_interest_change_z_7d_bucket"] == 5.0,
        ),
        (
            "veto_long_crowded_q5q5_or_short_mark_spread_q1",
            lambda row: (
                row["side"] == "LONG"
                and row["funding_rate_z_7d_bucket"] == 5.0
                and row["basis_proxy_bps_z_7d_bucket"] == 5.0
            )
            or (
                row["side"] == "SHORT"
                and row["mark_index_spread_bps_z_7d_bucket"] == 1.0
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for window, window_frame in enriched.groupby("window"):
        cohorts = {
            "all_trades": window_frame,
            "covered_only": window_frame.dropna(subset=FEATURES),
        }
        for cohort_name, cohort_frame in cohorts.items():
            baseline = summarize_subset(cohort_frame)
            rows.append(
                {
                    "window": window,
                    "cohort": cohort_name,
                    "candidate": "baseline_no_gate",
                    "vetoed_trades": 0,
                    "vetoed_r": 0.0,
                    **baseline,
                }
            )
            for label, rule in candidates:
                rule_mask = cohort_frame.apply(rule, axis=1)
                filtered = cohort_frame[~rule_mask].copy()
                vetoed = cohort_frame[rule_mask].copy()
                summary = summarize_subset(filtered)
                rows.append(
                    {
                        "window": window,
                        "cohort": cohort_name,
                        "candidate": label,
                        "vetoed_trades": int(len(vetoed)),
                        "vetoed_r": round(float(vetoed["pnl_r"].sum()), 4) if not vetoed.empty else 0.0,
                        **summary,
                    }
                )
    return rows


def build_coverage_rows(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window, frame in enriched.groupby("window"):
        joined = frame[FEATURES].notna().all(axis=1)
        rows.append(
            {
                "window": window,
                "trades": int(len(frame)),
                "joined_trades": int(joined.sum()),
                "join_rate_pct": round(float(joined.mean() * 100.0), 2),
            }
        )
    return rows


def build_report(coverage_rows: list[dict[str, Any]], top_bucket_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Mainline Derivatives Gate Study",
            "",
            "- 这轮先做 trade-filter 诊断，不是正式把 derivatives gate 接进撮合回测。",
            "- `mainline` 仍然是 `swing_trend_long_regime_gate_v1` + `LONG 1R -> 3R scaled / SHORT 1.5R fixed` + `confirmed swing`。",
            "- 衍生品状态来自 Bybit 小时面板，按 `signal_time` 对齐；因此这份结果只能回答“哪些状态下这笔交易更差/更好”，还不能回答“gated 后是否会释放新的后续机会”。",
            "",
            "## Join Coverage",
            "",
            markdown_table(
                coverage_rows,
                [
                    ("window", "Window"),
                    ("trades", "Trades"),
                    ("joined_trades", "Joined Trades"),
                    ("join_rate_pct", "Join Rate %"),
                ],
            ),
            "",
            "## Bucket Diagnostics",
            "",
            markdown_table(
                top_bucket_rows,
                [
                    ("window", "Window"),
                    ("side", "Side"),
                    ("feature", "Feature"),
                    ("bucket", "Bucket"),
                    ("feature_mean", "Feature Mean"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win Rate %"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                ],
            ),
            "",
            "## Naive Candidate Filters",
            "",
            markdown_table(
                candidate_rows,
                [
                    ("window", "Window"),
                    ("cohort", "Cohort"),
                    ("candidate", "Candidate"),
                    ("trades", "Trades"),
                    ("vetoed_trades", "Vetoed"),
                    ("vetoed_r", "Vetoed R"),
                    ("win_rate_pct", "Win Rate %"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                ],
            ),
            "",
        ]
    )


def main() -> None:
    configure_logging()
    ensure_output_dir()

    research = load_research_table()
    quantile_edges = build_quantile_edges(research)
    (OUTPUT_DIR / "quantile_edges.json").write_text(json.dumps(quantile_edges, indent=2), encoding="utf-8")

    window_trade_rows: list[dict[str, Any]] = []
    enriched_frames: list[pd.DataFrame] = []
    for window in WINDOWS:
        report = run_mainline(window)
        report_path = OUTPUT_DIR / f"{window}_{PROFILE}_report.json"
        trades_path = OUTPUT_DIR / f"{window}_{PROFILE}_trades.csv"
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        frame = trades_frame(report)
        frame.to_csv(trades_path, index=False)
        window_trade_rows.extend(frame.assign(window=window).to_dict("records"))
        enriched_frames.append(join_trades_with_derivatives(window=window, trades=frame, research=research, quantile_edges=quantile_edges))

    enriched = pd.concat(enriched_frames, ignore_index=True)
    enriched.to_csv(OUTPUT_DIR / "mainline_trades_with_derivatives.csv", index=False)

    coverage_rows = build_coverage_rows(enriched)
    bucket_rows = build_bucket_rows(enriched)
    write_csv(OUTPUT_DIR / "bucket_diagnostics.csv", bucket_rows)

    # Keep only the strongest negative/positive buckets for fast review.
    bucket_frame = pd.DataFrame(bucket_rows)
    top_bucket_rows: list[dict[str, Any]] = []
    if not bucket_frame.empty:
        bucket_frame = bucket_frame[bucket_frame["trades"] >= 5]
        top_negative = (
            bucket_frame.sort_values(["window", "expectancy_r"], ascending=[True, True])
            .groupby("window", group_keys=False)
            .head(6)
        )
        top_positive = (
            bucket_frame.sort_values(["window", "expectancy_r"], ascending=[True, False])
            .groupby("window", group_keys=False)
            .head(6)
        )
        top_bucket_rows = (
            pd.concat([top_negative, top_positive], ignore_index=True)
            .drop_duplicates(subset=["window", "side", "feature", "bucket"])
            .sort_values(["window", "expectancy_r"])
            .to_dict("records")
        )

    candidate_rows = build_candidate_rows(enriched)
    write_csv(OUTPUT_DIR / "candidate_filters.csv", candidate_rows)

    report_text = build_report(coverage_rows, top_bucket_rows, candidate_rows)
    (OUTPUT_DIR / "report.md").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
