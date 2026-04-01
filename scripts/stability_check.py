from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.backtesting.service import BacktestAssumptions, BacktestReport, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a stability check across calendar and rolling windows.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--strategy-profile", default="swing_trend_simple_candidate_v2")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default="2024-03-19")
    parser.add_argument("--end", default="2026-03-19")
    parser.add_argument("--output-dir", default="artifacts/backtests/btc_simple_candidates/stability")
    parser.add_argument("--rolling-180-step-days", type=int, default=30)
    parser.add_argument("--rolling-365-step-days", type=int, default=60)
    parser.add_argument(
        "--long-exit-json",
        default='{"take_profit_mode":"scaled","scaled_tp1_r":1.0,"scaled_tp2_r":3.0}',
    )
    parser.add_argument(
        "--short-exit-json",
        default='{"take_profit_mode":"fixed_r","fixed_take_profit_r":1.5}',
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def parse_exit_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Exit override must decode to a JSON object")
    return parsed


def build_service(args: argparse.Namespace) -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(
            exit_profile="stability_check_long_scaled1_3_short_fixed1_5",
            take_profit_mode="scaled",
            scaled_tp1_r=1.0,
            scaled_tp2_r=3.0,
            long_exit=parse_exit_json(args.long_exit_json),
            short_exit=parse_exit_json(args.short_exit_json),
        ),
    )


def trades_frame(report: BacktestReport) -> pd.DataFrame:
    trades = pd.DataFrame([asdict(trade) for trade in report.trades])
    if trades.empty:
        return trades
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    return trades


def period_start_for_half_year(ts: pd.Timestamp) -> pd.Timestamp:
    month = 1 if ts.month <= 6 else 7
    return pd.Timestamp(year=ts.year, month=month, day=1, tz="UTC")


def next_half_year_start(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.month <= 6:
        return pd.Timestamp(year=ts.year, month=7, day=1, tz="UTC")
    return pd.Timestamp(year=ts.year + 1, month=1, day=1, tz="UTC")


def iter_calendar_windows(start: datetime, end: datetime, mode: str) -> list[tuple[str, datetime, datetime]]:
    windows: list[tuple[str, datetime, datetime]] = []
    cursor = pd.Timestamp(start)
    limit = pd.Timestamp(end)

    while cursor < limit:
        if mode == "quarter":
            naive_cursor = cursor.tz_localize(None)
            period = naive_cursor.to_period("Q")
            period_start = period.start_time.tz_localize("UTC")
            next_start = (period + 1).start_time.tz_localize("UTC")
            label = f"{period.start_time.year}Q{period.quarter}"
        elif mode == "half_year":
            period_start = period_start_for_half_year(cursor)
            next_start = next_half_year_start(cursor)
            label = f"{period_start.year}H{1 if period_start.month == 1 else 2}"
        elif mode == "year":
            period_start = pd.Timestamp(year=cursor.year, month=1, day=1, tz="UTC")
            next_start = pd.Timestamp(year=cursor.year + 1, month=1, day=1, tz="UTC")
            label = str(cursor.year)
        else:
            raise ValueError(f"Unsupported calendar mode: {mode}")

        window_start = max(period_start.to_pydatetime(), start)
        window_end = min(next_start.to_pydatetime(), end)
        if window_start < window_end:
            windows.append((label, window_start, window_end))
        cursor = next_start

    return windows


def iter_rolling_windows(
    start: datetime,
    end: datetime,
    *,
    window_days: int,
    step_days: int,
) -> list[tuple[str, datetime, datetime]]:
    windows: list[tuple[str, datetime, datetime]] = []
    cursor = start
    index = 1
    while cursor + timedelta(days=window_days) <= end:
        window_end = cursor + timedelta(days=window_days)
        label = f"{window_days}d_{index:02d}"
        windows.append((label, cursor, window_end))
        cursor += timedelta(days=step_days)
        index += 1
    return windows


def summarize_trade_slice(
    trades: pd.DataFrame,
    *,
    label: str,
    horizon: str,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    window_trades = trades[
        (trades["entry_time"] >= pd.Timestamp(start))
        & (trades["entry_time"] < pd.Timestamp(end))
    ].copy()
    long_trades = window_trades[window_trades["side"] == "LONG"] if not window_trades.empty else pd.DataFrame()
    short_trades = window_trades[window_trades["side"] == "SHORT"] if not window_trades.empty else pd.DataFrame()

    total_trades = int(len(window_trades))
    cumulative_r = float(window_trades["pnl_r"].sum()) if total_trades else 0.0
    expectancy_r = float(window_trades["pnl_r"].mean()) if total_trades else 0.0
    gross_profit = float(window_trades.loc[window_trades["pnl_r"] > 0, "pnl_r"].sum()) if total_trades else 0.0
    gross_loss = float(window_trades.loc[window_trades["pnl_r"] < 0, "pnl_r"].sum()) if total_trades else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit > 0 else 0.0)
    win_rate = float((window_trades["pnl_r"] > 0).mean() * 100) if total_trades else 0.0
    avg_holding_bars = float(window_trades["bars_held"].mean()) if total_trades else 0.0
    tp1_hit_rate = float(window_trades["tp1_hit"].mean() * 100) if total_trades else 0.0
    tp2_hit_rate = float(window_trades["tp2_hit"].mean() * 100) if total_trades else 0.0
    if total_trades:
        equity = window_trades["pnl_r"].cumsum()
        drawdown = equity.cummax() - equity
        max_drawdown_r = float(drawdown.max())
    else:
        max_drawdown_r = 0.0

    return {
        "horizon": horizon,
        "label": label,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "days": int((end - start).days),
        "trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy_r,
        "cumulative_r": cumulative_r,
        "max_drawdown_r": max_drawdown_r,
        "avg_holding_bars": avg_holding_bars,
        "tp1_hit_rate": tp1_hit_rate,
        "tp2_hit_rate": tp2_hit_rate,
        "long_trades": int(len(long_trades)),
        "long_r": float(long_trades["pnl_r"].sum()) if not long_trades.empty else 0.0,
        "short_trades": int(len(short_trades)),
        "short_r": float(short_trades["pnl_r"].sum()) if not short_trades.empty else 0.0,
    }


def summarize_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "windows": 0,
            "positive_ratio": 0.0,
            "negative_count": 0,
            "median_expectancy_r": 0.0,
            "median_cumulative_r": 0.0,
            "best_label": "n/a",
            "best_cumulative_r": 0.0,
            "worst_label": "n/a",
            "worst_cumulative_r": 0.0,
        }
    ordered = sorted(rows, key=lambda item: item["cumulative_r"])
    positive = sum(1 for item in rows if item["cumulative_r"] > 0)
    return {
        "windows": len(rows),
        "positive_ratio": positive / len(rows),
        "negative_count": sum(1 for item in rows if item["cumulative_r"] <= 0),
        "median_expectancy_r": float(median(item["expectancy_r"] for item in rows)),
        "median_cumulative_r": float(median(item["cumulative_r"] for item in rows)),
        "best_label": ordered[-1]["label"],
        "best_cumulative_r": float(ordered[-1]["cumulative_r"]),
        "worst_label": ordered[0]["label"],
        "worst_cumulative_r": float(ordered[0]["cumulative_r"]),
    }


def full_trade_breakdown(report: BacktestReport) -> dict[str, pd.DataFrame]:
    trades = trades_frame(report)
    if trades.empty:
        empty = pd.DataFrame(columns=["bucket", "side", "trades", "cumulative_r", "avg_r", "win_rate"])
        return {"year": empty, "quarter": empty}

    trades["year"] = trades["entry_time"].dt.year.astype(str)
    trades["quarter"] = trades["entry_time"].dt.tz_localize(None).dt.to_period("Q").astype(str)

    def summarize(group_key: str) -> pd.DataFrame:
        grouped = (
            trades.groupby([group_key, "side"], as_index=False)
            .agg(
                trades=("pnl_r", "size"),
                cumulative_r=("pnl_r", "sum"),
                avg_r=("pnl_r", "mean"),
                win_rate=("pnl_r", lambda s: float((s > 0).mean() * 100)),
            )
            .rename(columns={group_key: "bucket"})
        )
        return grouped.sort_values(["bucket", "side"]).reset_index(drop=True)

    return {"year": summarize("year"), "quarter": summarize("quarter")}


def format_table(rows: list[dict[str, Any]], headers: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    head = "| " + " | ".join(label for _, label in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(row[key]) for key, _ in headers) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body])


def to_display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        display_rows.append(
            {
                "label": row["label"],
                "start": row["start"],
                "end": row["end"],
                "trades": row["trades"],
                "pf": f'{row["profit_factor"]:.2f}',
                "exp_r": f'{row["expectancy_r"]:.3f}',
                "cum_r": f'{row["cumulative_r"]:.2f}',
                "dd_r": f'{row["max_drawdown_r"]:.2f}',
                "long_r": f'{row["long_r"]:.2f}',
                "short_r": f'{row["short_r"]:.2f}',
            }
        )
    return display_rows


def trade_bucket_display(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        rows.append(
            {
                "bucket": item["bucket"],
                "side": item["side"],
                "trades": int(item["trades"]),
                "cum_r": f'{float(item["cumulative_r"]):.2f}',
                "avg_r": f'{float(item["avg_r"]):.3f}',
                "win_rate": f'{float(item["win_rate"]):.1f}%',
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    configure_logging()

    start = parse_date(args.start)
    end = parse_date(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    service = build_service(args)

    full_report = service.run(
        exchange=args.exchange,
        market_type=args.market_type,
        symbols=[args.symbol],
        strategy_profiles=[args.strategy_profile],
        start=start,
        end=end,
    )
    full_summary = full_report.overall[0]
    full_trades = trades_frame(full_report)

    window_rows: list[dict[str, Any]] = []
    for horizon, windows in (
        ("year", iter_calendar_windows(start, end, "year")),
        ("half_year", iter_calendar_windows(start, end, "half_year")),
        ("quarter", iter_calendar_windows(start, end, "quarter")),
        (
            "rolling_180d",
            iter_rolling_windows(start, end, window_days=180, step_days=args.rolling_180_step_days),
        ),
        (
            "rolling_365d",
            iter_rolling_windows(start, end, window_days=365, step_days=args.rolling_365_step_days),
        ),
    ):
        for label, window_start, window_end in windows:
            window_rows.append(
                summarize_trade_slice(
                    full_trades,
                    label=label,
                    horizon=horizon,
                    start=window_start,
                    end=window_end,
                )
            )

    window_csv = output_dir / f"stability_windows_{timestamp}.csv"
    with window_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(window_rows)

    trade_breakdown = full_trade_breakdown(full_report)
    summary_by_horizon = {
        horizon: summarize_windows([row for row in window_rows if row["horizon"] == horizon])
        for horizon in ("year", "half_year", "quarter", "rolling_180d", "rolling_365d")
    }

    markdown_path = output_dir / f"stability_check_{timestamp}.md"
    year_rows = [row for row in window_rows if row["horizon"] == "year"]
    half_year_rows = [row for row in window_rows if row["horizon"] == "half_year"]
    quarter_rows = [row for row in window_rows if row["horizon"] == "quarter"]
    rolling_180_rows = [row for row in window_rows if row["horizon"] == "rolling_180d"]
    rolling_365_rows = [row for row in window_rows if row["horizon"] == "rolling_365d"]

    top_quarter = trade_breakdown["quarter"].groupby("bucket", as_index=False)["cumulative_r"].sum()
    top_quarter = top_quarter.sort_values("cumulative_r", ascending=False).reset_index(drop=True)
    top_quarter_label = "n/a"
    top_quarter_share = 0.0
    if not top_quarter.empty and float(full_summary.cumulative_r) != 0:
        top_quarter_label = str(top_quarter.iloc[0]["bucket"])
        top_quarter_share = float(top_quarter.iloc[0]["cumulative_r"] / float(full_summary.cumulative_r))

    markdown = f"""# Stability Check

生成时间：{datetime.now(timezone.utc).isoformat()}

## 范围

- 标的：`{args.symbol}`
- 策略：`{args.strategy_profile}`
- 窗口：`{args.start} -> {args.end}`
- Exit：`LONG scaled 1R -> 3R, TP1 后保本；SHORT fixed 1.5R`
- 方法：`先跑一份全窗口正式回测，再按交易 entry_time 做年份 / 半年 / 季度 / 滚动窗口切片`

这里的结论适合做“主线稳定性体检”，但它不是严格的 walk-forward 重新回放。  
如果后续要做研究级别的 walk-forward，应单独实现复用同一份 OHLCV 的窗口引擎，而不是重复拉行情重跑。

## 全窗口结果

- 交易数：`{full_summary.total_trades}`
- Profit Factor：`{full_summary.profit_factor:.4f}`
- Expectancy：`{full_summary.expectancy_r:.4f}R`
- 累计：`{full_summary.cumulative_r:.4f}R`
- 最大回撤：`{full_summary.max_drawdown_r:.4f}R`
- 胜率：`{full_summary.win_rate:.2f}%`
- Top quarter 贡献占比：`{top_quarter_label} / {top_quarter_share:.2%}`

## 年度窗口

{format_table(
    to_display_rows(year_rows),
    [
        ("label", "窗口"),
        ("start", "开始"),
        ("end", "结束"),
        ("trades", "交易数"),
        ("pf", "PF"),
        ("exp_r", "Expectancy"),
        ("cum_r", "累计R"),
        ("dd_r", "回撤R"),
        ("long_r", "Long R"),
        ("short_r", "Short R"),
    ],
)}

年度摘要：

- 正收益窗口占比：`{summary_by_horizon["year"]["positive_ratio"]:.2%}`
- 中位 Expectancy：`{summary_by_horizon["year"]["median_expectancy_r"]:.4f}R`
- 最差年度：`{summary_by_horizon["year"]["worst_label"]} / {summary_by_horizon["year"]["worst_cumulative_r"]:.2f}R`
- 最佳年度：`{summary_by_horizon["year"]["best_label"]} / {summary_by_horizon["year"]["best_cumulative_r"]:.2f}R`

## 半年窗口

{format_table(
    to_display_rows(half_year_rows),
    [
        ("label", "窗口"),
        ("start", "开始"),
        ("end", "结束"),
        ("trades", "交易数"),
        ("pf", "PF"),
        ("exp_r", "Expectancy"),
        ("cum_r", "累计R"),
        ("dd_r", "回撤R"),
        ("long_r", "Long R"),
        ("short_r", "Short R"),
    ],
)}

半年摘要：

- 正收益窗口占比：`{summary_by_horizon["half_year"]["positive_ratio"]:.2%}`
- 中位 Expectancy：`{summary_by_horizon["half_year"]["median_expectancy_r"]:.4f}R`
- 最差半年：`{summary_by_horizon["half_year"]["worst_label"]} / {summary_by_horizon["half_year"]["worst_cumulative_r"]:.2f}R`
- 最佳半年：`{summary_by_horizon["half_year"]["best_label"]} / {summary_by_horizon["half_year"]["best_cumulative_r"]:.2f}R`

## 季度窗口

{format_table(
    to_display_rows(quarter_rows),
    [
        ("label", "窗口"),
        ("start", "开始"),
        ("end", "结束"),
        ("trades", "交易数"),
        ("pf", "PF"),
        ("exp_r", "Expectancy"),
        ("cum_r", "累计R"),
        ("dd_r", "回撤R"),
        ("long_r", "Long R"),
        ("short_r", "Short R"),
    ],
)}

季度摘要：

- 正收益窗口占比：`{summary_by_horizon["quarter"]["positive_ratio"]:.2%}`
- 中位 Expectancy：`{summary_by_horizon["quarter"]["median_expectancy_r"]:.4f}R`
- 最差季度：`{summary_by_horizon["quarter"]["worst_label"]} / {summary_by_horizon["quarter"]["worst_cumulative_r"]:.2f}R`
- 最佳季度：`{summary_by_horizon["quarter"]["best_label"]} / {summary_by_horizon["quarter"]["best_cumulative_r"]:.2f}R`

## 滚动 180 天

- 窗口数：`{summary_by_horizon["rolling_180d"]["windows"]}`
- 正收益窗口占比：`{summary_by_horizon["rolling_180d"]["positive_ratio"]:.2%}`
- 中位 Expectancy：`{summary_by_horizon["rolling_180d"]["median_expectancy_r"]:.4f}R`
- 中位累计：`{summary_by_horizon["rolling_180d"]["median_cumulative_r"]:.2f}R`
- 最差窗口：`{summary_by_horizon["rolling_180d"]["worst_label"]} / {summary_by_horizon["rolling_180d"]["worst_cumulative_r"]:.2f}R`
- 最佳窗口：`{summary_by_horizon["rolling_180d"]["best_label"]} / {summary_by_horizon["rolling_180d"]["best_cumulative_r"]:.2f}R`

## 滚动 365 天

- 窗口数：`{summary_by_horizon["rolling_365d"]["windows"]}`
- 正收益窗口占比：`{summary_by_horizon["rolling_365d"]["positive_ratio"]:.2%}`
- 中位 Expectancy：`{summary_by_horizon["rolling_365d"]["median_expectancy_r"]:.4f}R`
- 中位累计：`{summary_by_horizon["rolling_365d"]["median_cumulative_r"]:.2f}R`
- 最差窗口：`{summary_by_horizon["rolling_365d"]["worst_label"]} / {summary_by_horizon["rolling_365d"]["worst_cumulative_r"]:.2f}R`
- 最佳窗口：`{summary_by_horizon["rolling_365d"]["best_label"]} / {summary_by_horizon["rolling_365d"]["best_cumulative_r"]:.2f}R`

## 按 side 的全窗口分桶

### 按年

{format_table(
    trade_bucket_display(trade_breakdown["year"]),
    [
        ("bucket", "年份"),
        ("side", "方向"),
        ("trades", "交易数"),
        ("cum_r", "累计R"),
        ("avg_r", "平均R"),
        ("win_rate", "胜率"),
    ],
)}

### 按季度

{format_table(
    trade_bucket_display(trade_breakdown["quarter"]),
    [
        ("bucket", "季度"),
        ("side", "方向"),
        ("trades", "交易数"),
        ("cum_r", "累计R"),
        ("avg_r", "平均R"),
        ("win_rate", "胜率"),
    ],
)}

## 结论模板

- 如果年度和滚动 365 天经常翻负，就不能把它叫“稳定主线”。
- 如果季度起伏大，但年度和滚动 365 天大多为正，这更像可接受的波段系统。
- 如果 top quarter 贡献占比再次接近或超过总收益的大头，说明收益仍然集中，稳定性不足。

## 原始文件

- 窗口明细 CSV：`{window_csv}`
"""
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved markdown: {markdown_path}")
    print(f"Saved window CSV: {window_csv}")


if __name__ == "__main__":
    main()
