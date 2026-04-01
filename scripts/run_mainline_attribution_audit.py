from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
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
from app.services.mainline_attribution_audit_service import MainlineAttributionAuditService
from app.services.strategy_service import StrategyService


SYMBOL = "BTC/USDT:USDT"
EXCHANGE = "binance"
MARKET_TYPE = "perpetual"
PROFILE = "swing_trend_long_regime_gate_v1"
OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "mainline_attribution_audit"
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
TREND_BINS = [float("-inf"), 90.0, 94.0, 97.0, float("inf")]
TREND_LABELS = ["<=90", "91-94", "95-97", "98+"]
HOLD_BINS = [float("-inf"), 5.0, 15.0, 40.0, float("inf")]
HOLD_LABELS = ["<=5", "6-15", "16-40", "41+"]
VOLATILITY_ORDER = ["low", "normal", "high"]


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
        rendered: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
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


def build_trade_frame(report: BacktestReport, *, window_label: str) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(item) for item in report.trades])
    if frame.empty:
        return frame
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame["window_label"] = window_label
    return frame.sort_values("signal_time").reset_index(drop=True)


def resolve_index(frame: pd.DataFrame, timestamp: pd.Timestamp) -> int | None:
    left_idx = int(frame["timestamp"].searchsorted(timestamp, side="left"))
    if left_idx < len(frame) and frame.iloc[left_idx]["timestamp"] == timestamp:
        return left_idx
    right_idx = int(frame["timestamp"].searchsorted(timestamp, side="right") - 1)
    if right_idx >= 0 and frame.iloc[right_idx]["timestamp"] <= timestamp:
        return right_idx
    return None


def compute_trade_path_metrics(
    *,
    trigger_frame: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    side: str,
    entry_price: float,
    stop_price: float,
    pnl_r: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mfe_r": None,
        "mae_r": None,
        "mfe_capture_pct": None,
        "bars_held_actual": None,
    }
    risk_unit = abs(float(entry_price) - float(stop_price))
    if risk_unit <= 0:
        return row

    entry_idx = resolve_index(trigger_frame, entry_time)
    exit_idx = resolve_index(trigger_frame, exit_time)
    if entry_idx is None or exit_idx is None or exit_idx < entry_idx:
        return row

    path = trigger_frame.iloc[entry_idx : exit_idx + 1]
    path_high = float(path["high"].max())
    path_low = float(path["low"].min())

    if side == "LONG":
        mfe_r = max(0.0, (path_high - entry_price) / risk_unit)
        mae_r = max(0.0, (entry_price - path_low) / risk_unit)
    else:
        mfe_r = max(0.0, (entry_price - path_low) / risk_unit)
        mae_r = max(0.0, (path_high - entry_price) / risk_unit)

    row["mfe_r"] = round(float(mfe_r), 4)
    row["mae_r"] = round(float(mae_r), 4)
    row["bars_held_actual"] = int(len(path))
    if mfe_r > 0:
        row["mfe_capture_pct"] = round(float((pnl_r / mfe_r) * 100.0), 2)
    return row


def assign_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["year"] = pd.to_datetime(enriched["signal_time"], utc=True).dt.year.astype(int)
    enriched["trend_bucket"] = pd.cut(
        enriched["trend_strength"],
        bins=TREND_BINS,
        labels=TREND_LABELS,
        include_lowest=True,
    )
    enriched["hold_bucket"] = pd.cut(
        enriched["bars_held"],
        bins=HOLD_BINS,
        labels=HOLD_LABELS,
        include_lowest=True,
    )
    enriched["trend_bucket"] = enriched["trend_bucket"].astype(
        pd.CategoricalDtype(categories=TREND_LABELS, ordered=True)
    )
    enriched["hold_bucket"] = enriched["hold_bucket"].astype(
        pd.CategoricalDtype(categories=HOLD_LABELS, ordered=True)
    )
    enriched["volatility_state"] = enriched["volatility_state"].astype(
        pd.CategoricalDtype(categories=VOLATILITY_ORDER, ordered=True)
    )
    return enriched


def enrich_trades_with_context(
    *,
    service: BacktestService,
    trades: pd.DataFrame,
    start: datetime,
    history_end: datetime,
) -> pd.DataFrame:
    strategy = service.strategy_service.build_strategy(PROFILE)
    frames = service.prepare_history(
        exchange=EXCHANGE,
        market_type=MARKET_TYPE,
        symbol=SYMBOL,
        strategy_profile=PROFILE,
        start=start,
        end=history_end,
    )
    enriched_frames = service._prepare_enriched_frames(PROFILE, frames)
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    setup_tf = str(strategy.window_config["setup_timeframe"])
    trigger_frame = enriched_frames[trigger_tf]

    rows: list[dict[str, Any]] = []
    for trade in trades.to_dict(orient="records"):
        signal_time = pd.Timestamp(trade["signal_time"])
        indices: dict[str, int] = {}
        missing = False
        for timeframe in strategy.required_timeframes:
            idx = resolve_index(enriched_frames[timeframe], signal_time)
            if idx is None:
                missing = True
                break
            indices[timeframe] = idx
        if missing:
            continue

        prepared = {
            timeframe: service._build_snapshot(strategy, timeframe, enriched_frames[timeframe], indices[timeframe])
            for timeframe in indices
        }
        volatility_state = strategy._derive_volatility_state(prepared[setup_tf]).value
        path_metrics = compute_trade_path_metrics(
            trigger_frame=trigger_frame,
            entry_time=pd.Timestamp(trade["entry_time"]),
            exit_time=pd.Timestamp(trade["exit_time"]),
            side=str(trade["side"]),
            entry_price=float(trade["entry_price"]),
            stop_price=float(trade["stop_price"]),
            pnl_r=float(trade["pnl_r"]),
        )

        row = dict(trade)
        row["tp1_hit"] = bool(row["tp1_hit"])
        row["tp2_hit"] = bool(row["tp2_hit"])
        row["volatility_state"] = volatility_state
        row.update(path_metrics)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return assign_buckets(pd.DataFrame(rows)).sort_values(["window_label", "signal_time"]).reset_index(drop=True)


def run_window(service: BacktestService, *, window_label: str, start: datetime, end: datetime) -> tuple[BacktestReport, pd.DataFrame]:
    report = service.run(
        exchange=EXCHANGE,
        market_type=MARKET_TYPE,
        symbols=[SYMBOL],
        strategy_profiles=[PROFILE],
        start=start,
        end=end,
    )
    window_dir = OUTPUT_DIR / window_label
    BacktestService.save_report(report, window_dir)
    return report, build_trade_frame(report, window_label=window_label)


def render_report(
    *,
    overall_rows: list[dict[str, Any]],
    year_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    trend_rows: list[dict[str, Any]],
    volatility_rows: list[dict[str, Any]],
    hold_rows: list[dict[str, Any]],
    cluster_summary_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
) -> str:
    worst_clusters = sorted(cluster_rows, key=lambda item: (item["cluster_cumulative_r"], -item["cluster_length"]))[:6]
    lines = [
        "# Mainline Attribution Audit",
        "",
        f"- 生成时间: {datetime.now(timezone.utc).isoformat()}",
        f"- 标的: `{SYMBOL}`",
        f"- 策略: `{PROFILE}`",
        "- 口径: confirmed swing, LONG 1R -> 3R scaled, SHORT 1.5R fixed",
        "- 说明: MFE/MAE 基于实际持仓路径的价格极值计算, 不额外扣手续费与滑点",
        "",
        "## Overall",
        markdown_table(
            overall_rows,
            [
                ("window_label", "window"),
                ("total_trades", "trades"),
                ("win_rate", "win_rate"),
                ("profit_factor", "pf"),
                ("expectancy_r", "exp_r"),
                ("cumulative_r", "cum_r"),
                ("max_drawdown_r", "max_dd_r"),
                ("avg_holding_bars", "avg_bars"),
            ],
        ),
        "",
        "## By Year",
        markdown_table(
            year_rows,
            [
                ("window_label", "window"),
                ("year", "year"),
                ("trades", "trades"),
                ("win_rate_pct", "win_rate_pct"),
                ("expectancy_r", "exp_r"),
                ("cumulative_r", "cum_r"),
                ("profit_factor", "pf"),
                ("avg_mfe_r", "avg_mfe_r"),
                ("avg_mae_r", "avg_mae_r"),
            ],
        ),
        "",
        "## By Side",
        markdown_table(
            side_rows,
            [
                ("window_label", "window"),
                ("side", "side"),
                ("trades", "trades"),
                ("win_rate_pct", "win_rate_pct"),
                ("expectancy_r", "exp_r"),
                ("cumulative_r", "cum_r"),
                ("profit_factor", "pf"),
                ("avg_mfe_r", "avg_mfe_r"),
                ("avg_mae_r", "avg_mae_r"),
            ],
        ),
        "",
        "## By Trend Strength",
        markdown_table(
            trend_rows,
            [
                ("window_label", "window"),
                ("trend_bucket", "trend_bucket"),
                ("trades", "trades"),
                ("avg_trend_strength", "avg_ts"),
                ("expectancy_r", "exp_r"),
                ("cumulative_r", "cum_r"),
                ("profit_factor", "pf"),
                ("avg_mfe_r", "avg_mfe_r"),
                ("avg_mae_r", "avg_mae_r"),
            ],
        ),
        "",
        "## By Volatility State",
        markdown_table(
            volatility_rows,
            [
                ("window_label", "window"),
                ("volatility_state", "vol_state"),
                ("trades", "trades"),
                ("expectancy_r", "exp_r"),
                ("cumulative_r", "cum_r"),
                ("profit_factor", "pf"),
                ("avg_mfe_r", "avg_mfe_r"),
                ("avg_mae_r", "avg_mae_r"),
            ],
        ),
        "",
        "## By Hold Duration and MFE/MAE",
        markdown_table(
            hold_rows,
            [
                ("window_label", "window"),
                ("hold_bucket", "hold_bucket"),
                ("trades", "trades"),
                ("avg_bars_held", "avg_bars"),
                ("median_bars_held", "median_bars"),
                ("expectancy_r", "exp_r"),
                ("avg_mfe_r", "avg_mfe_r"),
                ("avg_mae_r", "avg_mae_r"),
                ("avg_mfe_capture_pct", "avg_capture_pct"),
            ],
        ),
        "",
        "## Loss Cluster Summary",
        markdown_table(
            cluster_summary_rows,
            [
                ("window_label", "window"),
                ("clusters", "clusters"),
                ("loss_trades_in_clusters", "loss_trades"),
                ("avg_cluster_length", "avg_len"),
                ("max_cluster_length", "max_len"),
                ("avg_cluster_cumulative_r", "avg_cluster_r"),
                ("worst_cluster_cumulative_r", "worst_cluster_r"),
                ("worst_cluster_length", "worst_len"),
            ],
        ),
        "",
        "## Worst Loss Clusters",
        markdown_table(
            worst_clusters,
            [
                ("window_label", "window"),
                ("cluster_id", "cluster_id"),
                ("cluster_start", "start"),
                ("cluster_end", "end"),
                ("cluster_length", "len"),
                ("cluster_cumulative_r", "cum_r"),
                ("avg_loss_r", "avg_loss_r"),
                ("worst_trade_r", "worst_trade_r"),
                ("long_count", "long_count"),
                ("short_count", "short_count"),
                ("years", "years"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    configure_logging()
    ensure_output_dir()
    service = make_backtest_service()
    audit_service = MainlineAttributionAuditService()

    reports: list[BacktestReport] = []
    trade_frames: list[pd.DataFrame] = []
    overall_rows: list[dict[str, Any]] = []

    for window_label, (start, end) in WINDOWS.items():
        report, trade_frame = run_window(service, window_label=window_label, start=start, end=end)
        reports.append(report)
        trade_frames.append(trade_frame)
        if report.overall:
            overall_row = asdict(report.overall[0])
            overall_row["window_label"] = window_label
            overall_rows.append(overall_row)

    trades = pd.concat(trade_frames, ignore_index=True).sort_values(["window_label", "signal_time"]).reset_index(drop=True)
    history_end = pd.to_datetime(trades["exit_time"], utc=True).max().to_pydatetime() + timedelta(hours=1)
    history_start = min(start for start, _ in WINDOWS.values())
    enriched_trades = enrich_trades_with_context(
        service=service,
        trades=trades,
        start=history_start,
        history_end=history_end,
    )

    trades_path = OUTPUT_DIR / "trades_enriched.csv"
    enriched_trades.to_csv(trades_path, index=False)

    year_rows = audit_service.summarize_groups(enriched_trades, group_cols=["window_label", "year"])
    side_rows = audit_service.summarize_groups(enriched_trades, group_cols=["window_label", "side"])
    trend_rows = audit_service.summarize_groups(enriched_trades, group_cols=["window_label", "trend_bucket"])
    volatility_rows = audit_service.summarize_groups(enriched_trades, group_cols=["window_label", "volatility_state"])
    hold_rows = audit_service.summarize_groups(enriched_trades, group_cols=["window_label", "hold_bucket"])
    cluster_rows = audit_service.identify_loss_clusters(enriched_trades)
    cluster_summary_rows = audit_service.summarize_loss_clusters(cluster_rows)

    write_csv(OUTPUT_DIR / "overall_summary.csv", overall_rows)
    write_csv(OUTPUT_DIR / "year_summary.csv", year_rows)
    write_csv(OUTPUT_DIR / "side_summary.csv", side_rows)
    write_csv(OUTPUT_DIR / "trend_strength_summary.csv", trend_rows)
    write_csv(OUTPUT_DIR / "volatility_summary.csv", volatility_rows)
    write_csv(OUTPUT_DIR / "hold_mfe_mae_summary.csv", hold_rows)
    write_csv(OUTPUT_DIR / "loss_clusters.csv", cluster_rows)
    write_csv(OUTPUT_DIR / "loss_cluster_summary.csv", cluster_summary_rows)

    report_text = render_report(
        overall_rows=overall_rows,
        year_rows=year_rows,
        side_rows=side_rows,
        trend_rows=trend_rows,
        volatility_rows=volatility_rows,
        hold_rows=hold_rows,
        cluster_summary_rows=cluster_summary_rows,
        cluster_rows=cluster_rows,
    )
    (OUTPUT_DIR / "report.md").write_text(report_text, encoding="utf-8")
    print(f"saved audit artifacts to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
