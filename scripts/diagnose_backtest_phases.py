from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.diagnostics import (
    bucket_confidence,
    bucket_distance,
    build_phase_funnel,
    collect_signal_diagnostics,
    derive_findings,
    summarize_performance,
)
from app.backtesting.service import BacktestAssumptions, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose stage-by-stage backtest behavior for one strategy/profile.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Unified ccxt symbol.")
    parser.add_argument("--strategy-profile", default="swing_trend_v1", help="Strategy profile.")
    parser.add_argument("--exchange", default="binance", help="Exchange id.")
    parser.add_argument("--market-type", default="perpetual", help="Market type.")
    parser.add_argument("--start", default=None, help="UTC start date, e.g. 2024-03-18")
    parser.add_argument("--end", default=None, help="UTC end date, e.g. 2026-03-18")
    parser.add_argument("--years", type=int, default=2, help="Fallback years lookback when --start is omitted.")
    parser.add_argument("--take-profit-mode", choices=["scaled", "fixed_r"], default="fixed_r")
    parser.add_argument("--fixed-take-profit-r", type=float, default=2.0)
    parser.add_argument("--scaled-tp1-r", type=float, default=None)
    parser.add_argument("--scaled-tp2-r", type=float, default=None)
    parser.add_argument("--long-exit-json", default=None, help="Optional LONG exit override JSON object.")
    parser.add_argument("--short-exit-json", default=None, help="Optional SHORT exit override JSON object.")
    parser.add_argument("--exit-profile", default="btc_swing_phase_diagnosis")
    parser.add_argument("--output-dir", default="artifacts/diagnostics", help="Output directory.")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def parse_exit_json(value: str | None, *, label: str) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return parsed


def slugify(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "default"


def render_markdown(report: dict[str, object]) -> str:
    headline = report["headline"]
    lines = [
        "# BTC 波段分阶段诊断",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 标的: {headline['symbol']}",
        f"- 策略: {headline['strategy_profile']}",
        f"- 回测窗口: {headline['start']} -> {headline['end']}",
        f"- 退出模型: {headline['exit_profile']}",
        "",
        "## 概览",
        "",
        f"- 交易数: {headline['total_trades']}",
        f"- 胜率: {headline['win_rate']}%",
        f"- 盈亏比: {headline['payoff_ratio']}",
        f"- Profit Factor: {headline['profit_factor']}",
        f"- 期望: {headline['expectancy_r']}R",
        f"- 累计: {headline['cumulative_r']}R",
        f"- 最大回撤: {headline['max_drawdown_r']}R",
        "",
        "## 关键发现",
        "",
    ]
    for item in report["key_findings"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 信号漏斗",
            "",
            "| 阶段 | 数量 | 占全部 | 占上一阶段 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in report["phase_funnel"]:
        lines.append(
            f"| {item['stage']} | {item['count']} | {item['pct_of_all']}% | {item['pct_of_prev']}% |"
        )

    lines.extend(
        [
            "",
            "## WAIT 主因",
            "",
            "| 原因 | 数量 |",
            "| --- | ---: |",
        ]
    )
    for item in report["wait_reason_breakdown"]:
        lines.append(f"| {item['wait_reason']} | {item['count']} |")

    section_specs = [
        ("方向分布", "by_side", "side"),
        ("退出原因", "by_exit_reason", "exit_reason"),
        ("季度表现", "by_quarter", "quarter"),
        ("年份表现", "by_year", "year"),
        ("置信度分桶", "by_confidence_bucket", "confidence_bucket"),
        ("入场位置分桶", "by_distance_bucket", "distance_bucket"),
    ]
    for title, key, column in section_specs:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"| {column} | count | win_rate | avg_r | cumulative_r | profit_factor |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in report[key]:
            lines.append(
                f"| {item[column]} | {item['count']} | {item['win_rate']}% | {item['avg_r']} | {item['cumulative_r']} | {item['profit_factor']} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    configure_logging()

    now = datetime.now(timezone.utc)
    end = parse_date(args.end) if args.end else now
    start = parse_date(args.start) if args.start else end - timedelta(days=args.years * 365)

    service = BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(
            exit_profile=args.exit_profile,
            take_profit_mode=args.take_profit_mode,
            fixed_take_profit_r=args.fixed_take_profit_r,
            scaled_tp1_r=args.scaled_tp1_r,
            scaled_tp2_r=args.scaled_tp2_r,
            long_exit=parse_exit_json(args.long_exit_json, label="--long-exit-json"),
            short_exit=parse_exit_json(args.short_exit_json, label="--short-exit-json"),
        ),
    )

    backtest_report = service.run(
        exchange=args.exchange,
        market_type=args.market_type,
        symbols=[args.symbol],
        strategy_profiles=[args.strategy_profile],
        start=start,
        end=end,
    )
    summary = backtest_report.by_symbol[0]

    signals = collect_signal_diagnostics(
        service=service,
        exchange=args.exchange,
        market_type=args.market_type,
        symbol=args.symbol,
        strategy_profile=args.strategy_profile,
        start=start,
        end=end,
    )
    trades = pd.DataFrame(
        [
            item.__dict__
            for item in backtest_report.trades
            if item.symbol == args.symbol and item.strategy_profile == args.strategy_profile
        ]
    )
    if trades.empty:
        trades = pd.DataFrame(columns=["signal_time", "side", "exit_reason", "confidence", "pnl_r", "bars_held"])

    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
    merged = trades.merge(signals, left_on="signal_time", right_on="timestamp", how="left")

    if not merged.empty:
        merged["entry_time"] = pd.to_datetime(merged["entry_time"], utc=True)
        merged["quarter"] = merged["entry_time"].dt.to_period("Q").astype(str)
        merged["year"] = merged["entry_time"].dt.year.astype(str)
        merged = merged.rename(columns={"confidence_x": "trade_confidence", "confidence_y": "signal_confidence"})
        merged["confidence_bucket"] = merged["trade_confidence"].astype(int).map(bucket_confidence)
        merged["distance_bucket"] = merged["setup_distance_to_execution_atr"].fillna(0.0).map(bucket_distance)
    else:
        merged["quarter"] = pd.Series(dtype=str)
        merged["year"] = pd.Series(dtype=str)
        merged["confidence_bucket"] = pd.Series(dtype=str)
        merged["distance_bucket"] = pd.Series(dtype=str)
        merged["trade_confidence"] = pd.Series(dtype=float)
        merged["signal_confidence"] = pd.Series(dtype=float)

    phase_funnel = build_phase_funnel(
        signals,
        action_threshold=int(service.strategy_service.build_strategy(args.strategy_profile).config["confidence"]["action_threshold"]),
    )
    wait_reason_breakdown = (
        signals.loc[~signals["signal_now"], "wait_reason"]
        .value_counts()
        .rename_axis("wait_reason")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "headline": {
            "symbol": args.symbol,
            "strategy_profile": args.strategy_profile,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "exit_profile": args.exit_profile,
            "total_trades": summary.total_trades,
            "win_rate": summary.win_rate,
            "payoff_ratio": summary.payoff_ratio,
            "profit_factor": summary.profit_factor,
            "expectancy_r": summary.expectancy_r,
            "cumulative_r": summary.cumulative_r,
            "max_drawdown_r": summary.max_drawdown_r,
        },
        "phase_funnel": phase_funnel,
        "wait_reason_breakdown": wait_reason_breakdown,
        "by_side": summarize_performance(merged, group_by="side"),
        "by_exit_reason": summarize_performance(merged, group_by="exit_reason"),
        "by_quarter": summarize_performance(merged, group_by="quarter"),
        "by_year": summarize_performance(merged, group_by="year"),
        "by_confidence_bucket": summarize_performance(merged, group_by="confidence_bucket"),
        "by_distance_bucket": summarize_performance(merged, group_by="distance_bucket"),
        "key_findings": derive_findings(
            summary=report_summary_to_dict(summary),
            trades=merged,
            phase_funnel=phase_funnel,
        ),
        "raw": {
            "signals_evaluated": int(len(signals)),
            "signals_now": int(signals["signal_now"].sum()) if not signals.empty else 0,
            "trades_opened": int(signals["entry_opened"].sum()) if not signals.empty else 0,
            "trades_skipped": int(signals["entry_skipped"].sum()) if not signals.empty else 0,
            "avg_confidence": round(float(merged["trade_confidence"].mean()) if not merged.empty else 0.0, 2),
        },
    }

    output_dir = Path(args.output_dir) / args.strategy_profile / "btc_phase_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    profile_slug = slugify(str(args.exit_profile))
    json_path = output_dir / f"diagnosis_{profile_slug}_{stamp}.json"
    md_path = output_dir / f"diagnosis_{profile_slug}_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Saved diagnosis JSON: {json_path}")
    print(f"Saved diagnosis Markdown: {md_path}")
    print(render_markdown(report))


def report_summary_to_dict(summary: object) -> dict[str, object]:
    return {
        "total_trades": getattr(summary, "total_trades"),
        "win_rate": getattr(summary, "win_rate"),
        "payoff_ratio": getattr(summary, "payoff_ratio"),
        "profit_factor": getattr(summary, "profit_factor"),
        "expectancy_r": getattr(summary, "expectancy_r"),
        "cumulative_r": getattr(summary, "cumulative_r"),
        "max_drawdown_r": getattr(summary, "max_drawdown_r"),
    }


if __name__ == "__main__":
    main()
