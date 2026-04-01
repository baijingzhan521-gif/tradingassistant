from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestTrade
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


EXIT_PRESETS: dict[str, dict[str, Any]] = {
    "long_scaled1_3_short_fixed1_5": {
        "exit_profile": "portfolio_long_scaled1_3_short_fixed1_5",
        "take_profit_mode": "scaled",
        "scaled_tp1_r": 1.0,
        "scaled_tp2_r": 3.0,
        "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    },
}

DEFAULT_SYMBOLS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "AAVE/USDT:USDT",
    "HYPE/USDT:USDT",
    "BNB/USDT:USDT",
)

DEFAULT_BASKETS = (
    "btc=BTC/USDT:USDT",
    "majors3=BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT",
    "alts5=ETH/USDT:USDT,SOL/USDT:USDT,AAVE/USDT:USDT,HYPE/USDT:USDT,BNB/USDT:USDT",
    "full6=BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,AAVE/USDT:USDT,HYPE/USDT:USDT,BNB/USDT:USDT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-asset portfolio walk-forward validation.")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default="2024-03-19")
    parser.add_argument("--end", default="2026-03-19")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--scheme", choices=["rolling", "anchored"], default="rolling")
    parser.add_argument("--strategy-profile", default="swing_trend_long_regime_gate_v1")
    parser.add_argument("--exit-preset", choices=sorted(EXIT_PRESETS.keys()), default="long_scaled1_3_short_fixed1_5")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--baskets",
        default=";".join(DEFAULT_BASKETS),
        help="Semicolon-separated baskets. Format: name=symbol1,symbol2",
    )
    parser.add_argument("--output-dir", default="artifacts/backtests/multi_asset_portfolio_walk_forward")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_service(exit_preset: str) -> BacktestService:
    assumptions = {
        **EXIT_PRESETS[exit_preset],
        "swing_detection_mode": "confirmed",
    }
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**assumptions),
    )


def parse_symbols(raw: str) -> list[str]:
    symbols = [item.strip() for item in raw.split(",") if item.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    return symbols


def parse_baskets(raw: str) -> dict[str, list[str]]:
    baskets: dict[str, list[str]] = {}
    for chunk in [item.strip() for item in raw.split(";") if item.strip()]:
        if "=" not in chunk:
            raise ValueError(f"Invalid basket spec: {chunk}")
        name, members_raw = chunk.split("=", 1)
        name = name.strip()
        members = [item.strip() for item in members_raw.split(",") if item.strip()]
        if not name or not members:
            raise ValueError(f"Invalid basket spec: {chunk}")
        baskets[name] = members
    if not baskets:
        raise ValueError("At least one basket is required")
    return baskets


def generate_folds(
    *,
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    scheme: str,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    anchor_start = start
    train_start = start
    train_end = train_start + timedelta(days=train_days)
    index = 1

    while train_end + timedelta(days=test_days) <= end:
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        folds.append(
            {
                "fold": index,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        index += 1
        if scheme == "anchored":
            train_end = train_end + timedelta(days=step_days)
            train_start = anchor_start
        else:
            train_start = train_start + timedelta(days=step_days)
            train_end = train_start + timedelta(days=train_days)

    return folds


def to_summary_dict(summary) -> dict[str, Any]:
    return {
        "total_trades": int(summary.total_trades),
        "win_rate": float(summary.win_rate),
        "profit_factor": float(summary.profit_factor),
        "expectancy_r": float(summary.expectancy_r),
        "cumulative_r": float(summary.cumulative_r),
        "max_drawdown_r": float(summary.max_drawdown_r),
        "avg_holding_bars": float(summary.avg_holding_bars),
        "tp1_hit_rate": float(summary.tp1_hit_rate),
        "tp2_hit_rate": float(summary.tp2_hit_rate),
        "signals_now": int(summary.signals_now),
        "skipped_entries": int(summary.skipped_entries),
    }


def format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row[key]) for key, _ in columns) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def sort_trades(trades: list[BacktestTrade]) -> list[BacktestTrade]:
    return sorted(trades, key=lambda item: (item.exit_time, item.entry_time, item.symbol, item.side))


def scale_trade(trade: BacktestTrade, factor: float) -> BacktestTrade:
    data = asdict(trade)
    data["pnl_r"] = round(float(data["pnl_r"]) * factor, 10)
    data["pnl_pct"] = round(float(data["pnl_pct"]) * factor, 10)
    data["gross_pnl_quote"] = round(float(data["gross_pnl_quote"]) * factor, 10)
    data["fees_quote"] = round(float(data["fees_quote"]) * factor, 10)
    return BacktestTrade(**data)


def summarize_trades(
    service: BacktestService,
    *,
    trades: list[BacktestTrade],
    strategy_profile: str,
    label: str,
) -> dict[str, Any]:
    summary = service._summarize_trades(
        trades=sort_trades(trades),
        strategy_profile=strategy_profile,
        symbol=label,
        signals_now=0,
        skipped_entries=0,
    )
    return to_summary_dict(summary)


def positive_fold_ratio(fold_summaries: list[dict[str, Any]]) -> float:
    if not fold_summaries:
        return 0.0
    return sum(1 for item in fold_summaries if float(item["cumulative_r"]) > 0.0) / len(fold_summaries)


def main() -> None:
    args = parse_args()
    configure_logging()

    start = parse_date(args.start)
    end = parse_date(args.end)
    symbols = parse_symbols(args.symbols)
    baskets = parse_baskets(args.baskets)
    folds = generate_folds(
        start=start,
        end=end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        scheme=args.scheme,
    )
    if not folds:
        raise ValueError("No valid folds. Reduce train/test length or expand the overall window.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    service = build_service(args.exit_preset)

    enriched_history: dict[str, dict[str, pd.DataFrame]] = {}
    symbol_errors: dict[str, str] = {}
    for symbol in symbols:
        print(f"[portfolio] preload {symbol}", flush=True)
        try:
            enriched_history[symbol] = service.prepare_enriched_history(
                exchange=args.exchange,
                market_type=args.market_type,
                symbol=symbol,
                strategy_profile=args.strategy_profile,
                start=start,
                end=end,
            )
            print(f"[portfolio] ready {symbol}", flush=True)
        except Exception as exc:  # noqa: BLE001
            symbol_errors[symbol] = f"{type(exc).__name__}: {exc}"
            print(f"[portfolio] skip {symbol}: {symbol_errors[symbol]}", flush=True)

    available_symbols = [symbol for symbol in symbols if symbol in enriched_history]
    if "BTC/USDT:USDT" not in available_symbols:
        raise RuntimeError("BTC baseline is unavailable, cannot compare portfolio route fairly.")

    basket_status_rows: list[dict[str, Any]] = []
    valid_baskets: dict[str, list[str]] = {}
    for name, members in baskets.items():
        missing = [symbol for symbol in members if symbol not in available_symbols]
        if missing:
            basket_status_rows.append(
                {
                    "basket": name,
                    "symbols": ", ".join(members),
                    "status": f"skip (missing: {', '.join(missing)})",
                }
            )
            continue
        valid_baskets[name] = members
        basket_status_rows.append({"basket": name, "symbols": ", ".join(members), "status": "ok"})

    if "btc" not in valid_baskets:
        valid_baskets["btc"] = ["BTC/USDT:USDT"]
        basket_status_rows.append({"basket": "btc", "symbols": "BTC/USDT:USDT", "status": "auto-added"})

    per_symbol_fold_rows: list[dict[str, Any]] = []
    per_symbol_oos_trades: dict[str, list[BacktestTrade]] = {symbol: [] for symbol in available_symbols}
    basket_fold_rows: list[dict[str, Any]] = []
    basket_oos_trade_rows: list[dict[str, Any]] = []
    basket_oos_trades: dict[str, list[BacktestTrade]] = {name: [] for name in valid_baskets}
    basket_scaled_oos_trades: dict[str, list[BacktestTrade]] = {name: [] for name in valid_baskets}

    for fold in folds:
        print(
            f"[portfolio] fold {fold['fold']}/{len(folds)} train {fold['train_start'].date()}->{fold['train_end'].date()} test {fold['test_start'].date()}->{fold['test_end'].date()}",
            flush=True,
        )
        fold_oos_trades_by_symbol: dict[str, list[BacktestTrade]] = {}

        for symbol in available_symbols:
            train_summary, _ = service.run_symbol_strategy_with_enriched_frames(
                symbol=symbol,
                strategy_profile=args.strategy_profile,
                start=fold["train_start"],
                end=fold["train_end"],
                enriched_frames=enriched_history[symbol],
            )
            oos_summary, oos_trades = service.run_symbol_strategy_with_enriched_frames(
                symbol=symbol,
                strategy_profile=args.strategy_profile,
                start=fold["test_start"],
                end=fold["test_end"],
                enriched_frames=enriched_history[symbol],
            )
            train_dict = to_summary_dict(train_summary)
            oos_dict = to_summary_dict(oos_summary)
            per_symbol_fold_rows.append(
                {
                    "fold": fold["fold"],
                    "symbol": symbol,
                    "train_start": fold["train_start"].date().isoformat(),
                    "train_end": fold["train_end"].date().isoformat(),
                    "test_start": fold["test_start"].date().isoformat(),
                    "test_end": fold["test_end"].date().isoformat(),
                    **{f"train_{key}": value for key, value in train_dict.items()},
                    **{f"oos_{key}": value for key, value in oos_dict.items()},
                }
            )
            fold_oos_trades_by_symbol[symbol] = sort_trades(oos_trades)
            per_symbol_oos_trades[symbol].extend(oos_trades)

        for basket_name, members in valid_baskets.items():
            raw_trades: list[BacktestTrade] = []
            for symbol in members:
                raw_trades.extend(fold_oos_trades_by_symbol.get(symbol, []))
            raw_trades = sort_trades(raw_trades)
            sleeve_factor = 1.0 / len(members)
            scaled_trades = [scale_trade(trade, sleeve_factor) for trade in raw_trades]
            raw_summary = summarize_trades(
                service,
                trades=raw_trades,
                strategy_profile=args.strategy_profile,
                label=f"{basket_name}_raw",
            )
            scaled_summary = summarize_trades(
                service,
                trades=scaled_trades,
                strategy_profile=args.strategy_profile,
                label=f"{basket_name}_equal_sleeve",
            )
            basket_fold_rows.append(
                {
                    "fold": fold["fold"],
                    "basket": basket_name,
                    "member_count": len(members),
                    "members": ",".join(members),
                    "test_start": fold["test_start"].date().isoformat(),
                    "test_end": fold["test_end"].date().isoformat(),
                    **{f"raw_{key}": value for key, value in raw_summary.items()},
                    **{f"scaled_{key}": value for key, value in scaled_summary.items()},
                }
            )
            basket_oos_trades[basket_name].extend(raw_trades)
            basket_scaled_oos_trades[basket_name].extend(scaled_trades)
            for trade in scaled_trades:
                row = asdict(trade)
                row["fold"] = fold["fold"]
                row["basket"] = basket_name
                row["basket_member_count"] = len(members)
                row["portfolio_mode"] = "equal_sleeve"
                basket_oos_trade_rows.append(row)

    symbol_summary_rows: list[dict[str, Any]] = []
    for symbol in available_symbols:
        summary = summarize_trades(
            service,
            trades=per_symbol_oos_trades[symbol],
            strategy_profile=args.strategy_profile,
            label=symbol,
        )
        fold_summaries = [row for row in per_symbol_fold_rows if row["symbol"] == symbol]
        symbol_summary_rows.append(
            {
                "symbol": symbol,
                "oos_trades": summary["total_trades"],
                "oos_pf": f'{summary["profit_factor"]:.2f}',
                "oos_exp": f'{summary["expectancy_r"]:.3f}',
                "oos_cum_r": f'{summary["cumulative_r"]:.2f}',
                "oos_max_dd": f'{summary["max_drawdown_r"]:.2f}',
                "positive_folds": f'{sum(1 for row in fold_summaries if float(row["oos_cumulative_r"]) > 0)}/{len(folds)}',
            }
        )

    basket_summary_rows: list[dict[str, Any]] = []
    basket_fold_display_rows: list[dict[str, Any]] = []
    btc_scaled_summary: dict[str, Any] | None = None
    for basket_name, members in valid_baskets.items():
        raw_summary = summarize_trades(
            service,
            trades=basket_oos_trades[basket_name],
            strategy_profile=args.strategy_profile,
            label=f"{basket_name}_raw",
        )
        scaled_summary = summarize_trades(
            service,
            trades=basket_scaled_oos_trades[basket_name],
            strategy_profile=args.strategy_profile,
            label=f"{basket_name}_equal_sleeve",
        )
        fold_summaries = [
            {
                "cumulative_r": row["scaled_cumulative_r"],
                "profit_factor": row["scaled_profit_factor"],
                "expectancy_r": row["scaled_expectancy_r"],
            }
            for row in basket_fold_rows
            if row["basket"] == basket_name
        ]
        if basket_name == "btc":
            btc_scaled_summary = scaled_summary
        basket_summary_rows.append(
            {
                "basket": basket_name,
                "members": len(members),
                "raw_cum_r": f'{raw_summary["cumulative_r"]:.2f}',
                "scaled_cum_r": f'{scaled_summary["cumulative_r"]:.2f}',
                "scaled_pf": f'{scaled_summary["profit_factor"]:.2f}',
                "scaled_exp": f'{scaled_summary["expectancy_r"]:.3f}',
                "scaled_max_dd": f'{scaled_summary["max_drawdown_r"]:.2f}',
                "positive_folds": f"{sum(1 for item in fold_summaries if float(item['cumulative_r']) > 0)}/{len(folds)}",
                "positive_fold_ratio": f"{positive_fold_ratio(fold_summaries):.0%}",
            }
        )
        for row in [item for item in basket_fold_rows if item["basket"] == basket_name]:
            basket_fold_display_rows.append(
                {
                    "fold": row["fold"],
                    "basket": basket_name,
                    "scaled_trades": row["scaled_total_trades"],
                    "scaled_pf": f'{float(row["scaled_profit_factor"]):.2f}',
                    "scaled_exp": f'{float(row["scaled_expectancy_r"]):.3f}',
                    "scaled_cum_r": f'{float(row["scaled_cumulative_r"]):.2f}',
                }
            )

    if btc_scaled_summary is not None:
        for row in basket_summary_rows:
            row["vs_btc_cum_r"] = f'{float(row["scaled_cum_r"]) - float(btc_scaled_summary["cumulative_r"]):+.2f}'
            row["vs_btc_max_dd"] = f'{float(row["scaled_max_dd"]) - float(btc_scaled_summary["max_drawdown_r"]):+.2f}'
    else:
        for row in basket_summary_rows:
            row["vs_btc_cum_r"] = "n/a"
            row["vs_btc_max_dd"] = "n/a"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exchange": args.exchange,
        "market_type": args.market_type,
        "start": args.start,
        "end": args.end,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "scheme": args.scheme,
        "strategy_profile": args.strategy_profile,
        "exit_preset": args.exit_preset,
        "swing_detection_mode": "confirmed",
        "requested_symbols": symbols,
        "available_symbols": available_symbols,
        "symbol_errors": symbol_errors,
        "baskets": valid_baskets,
        "per_symbol_folds": per_symbol_fold_rows,
        "basket_folds": basket_fold_rows,
        "symbol_oos_summary": symbol_summary_rows,
        "basket_oos_summary": basket_summary_rows,
    }

    json_path = output_dir / f"multi_asset_walk_forward_{timestamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    symbol_csv = output_dir / f"multi_asset_symbol_folds_{timestamp}.csv"
    pd.DataFrame(per_symbol_fold_rows).to_csv(symbol_csv, index=False)

    basket_csv = output_dir / f"multi_asset_basket_folds_{timestamp}.csv"
    pd.DataFrame(basket_fold_rows).to_csv(basket_csv, index=False)

    oos_trades_csv = output_dir / f"multi_asset_oos_equal_sleeve_trades_{timestamp}.csv"
    pd.DataFrame(basket_oos_trade_rows).to_csv(oos_trades_csv, index=False)

    markdown_path = output_dir / f"multi_asset_walk_forward_report_{timestamp}.md"
    markdown = f"""# 多标的组合 Walk-Forward

生成时间：{datetime.now(timezone.utc).isoformat()}

## 设定

- 策略：`{args.strategy_profile}`
- Exit：`{args.exit_preset}`
- Swing 模式：`confirmed`
- 总窗口：`{args.start} -> {args.end}`
- 训练窗：`{args.train_days}` 天
- 测试窗：`{args.test_days}` 天
- 步长：`{args.step_days}` 天
- 方案：`{args.scheme}`

这里刻意**不**为不同标的改规则，也**不**在 symbol 之间做参数差异化；测的是“同一条 BTC 主线复制到多标的后，组合层面是否仍更好”。

## 数据可用性

{format_table(basket_status_rows, [("basket", "篮子"), ("symbols", "成员"), ("status", "状态")])}

请求 universe：`{", ".join(symbols)}`

实际可用 universe：`{", ".join(available_symbols)}`

## 单标的 OOS

{format_table(
    symbol_summary_rows,
    [
        ("symbol", "标的"),
        ("oos_trades", "OOS 交易数"),
        ("oos_pf", "OOS PF"),
        ("oos_exp", "OOS Exp"),
        ("oos_cum_r", "OOS 累计R"),
        ("oos_max_dd", "OOS 最大回撤"),
        ("positive_folds", "正收益 Fold"),
    ],
)}

## 组合 OOS

下面同时给两个口径：

- `raw_cum_r`：直接把所有标的交易相加。这个值**不能**和 BTC-only 直接横比，因为它隐含用了更多资本。
- `scaled_*`：等 sleeve 归一化，每个标的只占组合的 `1/N` 风险预算。只有这个口径，才适合和 BTC-only 做公平比较。

{format_table(
    basket_summary_rows,
    [
        ("basket", "篮子"),
        ("members", "成员数"),
        ("raw_cum_r", "Raw 累计R"),
        ("scaled_cum_r", "等权累计R"),
        ("scaled_pf", "等权 PF"),
        ("scaled_exp", "等权 Exp"),
        ("scaled_max_dd", "等权最大回撤"),
        ("positive_folds", "正收益 Fold"),
        ("vs_btc_cum_r", "相对 BTC 累计R"),
        ("vs_btc_max_dd", "相对 BTC 回撤"),
    ],
)}

## Fold 级组合表现

{format_table(
    basket_fold_display_rows,
    [
        ("fold", "Fold"),
        ("basket", "篮子"),
        ("scaled_trades", "等权交易数"),
        ("scaled_pf", "等权 PF"),
        ("scaled_exp", "等权 Exp"),
        ("scaled_cum_r", "等权累计R"),
    ],
)}

## 解释边界

- 这不是“多资产择时系统”，而是“固定同一策略规则，验证复制后组合层是否更优”。
- `scaled_*` 用的是等 sleeve 归一化；它能解决“大篮子天然更大资本”的比较偏差，但仍然不是完整的逐时盯市组合回撤模型。
- 如果某些单标的本身 OOS 不能独立站住，那么把它硬塞进组合，得到的更高 `raw_cum_r` 没有意义。

## 原始文件

- JSON：`{json_path}`
- 单标的 fold CSV：`{symbol_csv}`
- 组合 fold CSV：`{basket_csv}`
- 等权 OOS trades CSV：`{oos_trades_csv}`
"""
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved markdown: {markdown_path}")
    print(f"Saved json: {json_path}")
    print(f"Saved symbol CSV: {symbol_csv}")
    print(f"Saved basket CSV: {basket_csv}")
    print(f"Saved OOS trades CSV: {oos_trades_csv}")


if __name__ == "__main__":
    main()
