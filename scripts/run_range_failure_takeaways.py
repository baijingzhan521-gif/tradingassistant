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
OUTPUT_DIR = ROOT / "artifacts/backtests/btc_range_failure_v1"
MAINLINE = "swing_trend_long_regime_gate_v1"
RANGE_FAILURE = "swing_range_failure_v1_btc"
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


def make_service(assumptions: BacktestAssumptions) -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )


def run_profile(window: str, profile: str) -> BacktestReport:
    start, end = WINDOWS[window]
    if profile == MAINLINE:
        assumptions = BacktestAssumptions(
            exit_profile="long_scaled1_3_short_fixed1_5",
            take_profit_mode="scaled",
            long_exit={"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
            short_exit={"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
            swing_detection_mode="confirmed",
            cache_dir="artifacts/backtests/cache",
        )
    else:
        assumptions = BacktestAssumptions(
            exit_profile="range_failure_strategy_defined",
            take_profit_mode="scaled",
            swing_detection_mode="confirmed",
            cache_dir="artifacts/backtests/cache",
        )
    service = make_service(assumptions)
    return service.run(
        exchange="binance",
        market_type="perpetual",
        symbols=[SYMBOL],
        strategy_profiles=[profile],
        start=start,
        end=end,
    )


def report_to_rows(window: str, report: BacktestReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.overall:
        rows.append(
            {
                "window": window,
                "strategy_profile": item.strategy_profile,
                "trades": item.total_trades,
                "win_rate_pct": round(item.win_rate, 2),
                "profit_factor": round(item.profit_factor, 4),
                "expectancy_r": round(item.expectancy_r, 4),
                "cum_r": round(item.cumulative_r, 4),
                "max_dd_r": round(item.max_drawdown_r, 4),
                "avg_holding_bars": round(item.avg_holding_bars, 2),
                "tp1_hit_rate_pct": round(item.tp1_hit_rate, 2),
                "tp2_hit_rate_pct": round(item.tp2_hit_rate, 2),
            }
        )
    return rows


def trades_frame(report: BacktestReport) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(item) for item in report.trades])
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame["signal_month"] = frame["signal_time"].dt.strftime("%Y-%m")
    frame["exit_month"] = frame["exit_time"].dt.strftime("%Y-%m")
    return frame


def summarize_complementarity(window: str, mainline_trades: pd.DataFrame, range_trades: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mainline_monthly = (
        mainline_trades.groupby("signal_month")["pnl_r"].sum().rename("mainline_r").reset_index()
    )
    range_monthly = (
        range_trades.groupby("signal_month")["pnl_r"].sum().rename("range_failure_r").reset_index()
    )
    monthly = mainline_monthly.merge(range_monthly, on="signal_month", how="outer").fillna(0.0)
    monthly["window"] = window
    monthly["combined_naive_r"] = monthly["mainline_r"] + monthly["range_failure_r"]
    monthly["mainline_negative_range_positive"] = (monthly["mainline_r"] < 0) & (monthly["range_failure_r"] > 0)
    monthly["opposite_sign"] = (monthly["mainline_r"] * monthly["range_failure_r"]) < 0
    monthly = monthly.sort_values("signal_month").reset_index(drop=True)

    corr = None
    if len(monthly) >= 2 and monthly["mainline_r"].std() > 0 and monthly["range_failure_r"].std() > 0:
        corr = float(monthly["mainline_r"].corr(monthly["range_failure_r"]))

    offset_months = monthly[monthly["mainline_negative_range_positive"]].copy()
    summary = {
        "window": window,
        "months": int(len(monthly)),
        "monthly_corr": round(corr, 4) if corr is not None else None,
        "mainline_negative_range_positive_count": int(offset_months.shape[0]),
        "mainline_negative_range_positive_r_sum": round(float(offset_months["range_failure_r"].sum()), 4),
        "opposite_sign_count": int(monthly["opposite_sign"].sum()),
        "naive_combined_r": round(float(monthly["combined_naive_r"].sum()), 4),
    }

    monthly_rows = monthly[
        [
            "window",
            "signal_month",
            "mainline_r",
            "range_failure_r",
            "combined_naive_r",
            "mainline_negative_range_positive",
            "opposite_sign",
        ]
    ].rename(columns={"signal_month": "month"}).to_dict("records")
    return summary, monthly_rows


def build_report(
    summary_rows: list[dict[str, Any]],
    complementarity_rows: list[dict[str, Any]],
    top_offset_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# BTC Range Failure V1 Takeaways",
            "",
            "- 这轮不是要把 `range-failure` 直接升级成主线，而是先验证它是否和当前主线属于不同 alpha 家族。",
            "- `mainline` 仍按当前主线口径回测：`LONG 1R -> 3R scaled`，`SHORT 1.5R fixed`，`confirmed swing`。",
            "- `range_failure` 使用策略自带的区间中轴 / 区间对侧目标；因此月度互补性只能看方向和分布，不能把 `combined_naive_r` 当真实组合收益。",
            "",
            "## Standalone Results",
            "",
            markdown_table(
                summary_rows,
                [
                    ("window", "Window"),
                    ("strategy_profile", "Profile"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win Rate %"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "DD R"),
                    ("avg_holding_bars", "Avg Bars"),
                ],
            ),
            "",
            "## Complementarity Summary",
            "",
            markdown_table(
                complementarity_rows,
                [
                    ("window", "Window"),
                    ("months", "Months"),
                    ("monthly_corr", "Monthly Corr"),
                    ("mainline_negative_range_positive_count", "Main Neg / Range Pos"),
                    ("mainline_negative_range_positive_r_sum", "Range R In Those Months"),
                    ("opposite_sign_count", "Opposite Sign Months"),
                    ("naive_combined_r", "Naive Combined R"),
                ],
            ),
            "",
            "## Offset Months",
            "",
            markdown_table(
                top_offset_rows,
                [
                    ("window", "Window"),
                    ("month", "Month"),
                    ("mainline_r", "Mainline R"),
                    ("range_failure_r", "Range Failure R"),
                    ("combined_naive_r", "Naive Combined R"),
                    ("mainline_negative_range_positive", "Main Neg / Range Pos"),
                    ("opposite_sign", "Opposite Sign"),
                ],
            ),
            "",
        ]
    )


def main() -> None:
    configure_logging()
    ensure_output_dir()

    all_summary_rows: list[dict[str, Any]] = []
    complementarity_rows: list[dict[str, Any]] = []
    top_offset_rows: list[dict[str, Any]] = []

    for window in WINDOWS:
        mainline_report = run_profile(window, MAINLINE)
        range_report = run_profile(window, RANGE_FAILURE)

        (OUTPUT_DIR / f"{window}_{MAINLINE}_report.json").write_text(
            json.dumps(mainline_report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (OUTPUT_DIR / f"{window}_{RANGE_FAILURE}_report.json").write_text(
            json.dumps(range_report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        mainline_trades = trades_frame(mainline_report)
        range_trades = trades_frame(range_report)
        mainline_trades.to_csv(OUTPUT_DIR / f"{window}_{MAINLINE}_trades.csv", index=False)
        range_trades.to_csv(OUTPUT_DIR / f"{window}_{RANGE_FAILURE}_trades.csv", index=False)

        all_summary_rows.extend(report_to_rows(window, mainline_report))
        all_summary_rows.extend(report_to_rows(window, range_report))

        complementarity_summary, monthly_rows = summarize_complementarity(window, mainline_trades, range_trades)
        complementarity_rows.append(complementarity_summary)
        top_offset_rows.extend([row for row in monthly_rows if row.get("mainline_negative_range_positive")])
        write_csv(OUTPUT_DIR / f"{window}_monthly_complementarity.csv", monthly_rows)

    write_csv(OUTPUT_DIR / "summary_all.csv", all_summary_rows)
    write_csv(OUTPUT_DIR / "complementarity_summary.csv", complementarity_rows)
    if top_offset_rows:
        write_csv(OUTPUT_DIR / "offset_months.csv", top_offset_rows)

    report = build_report(
        summary_rows=all_summary_rows,
        complementarity_rows=complementarity_rows,
        top_offset_rows=top_offset_rows,
    )
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
