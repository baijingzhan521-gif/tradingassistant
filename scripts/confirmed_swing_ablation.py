from __future__ import annotations

import argparse
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

from app.backtesting.service import BacktestAssumptions, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


DEFAULT_PROFILES = (
    "swing_trend_long_regime_gate_v1",
    "swing_trend_ablation_no_reversal_v1",
    "swing_trend_ablation_no_auxiliary_v1",
    "swing_trend_ablation_no_held_slow_v1",
    "swing_trend_ablation_no_regained_fast_v1",
    "swing_trend_ablation_minimal_trigger_v1",
    "swing_trend_ablation_symmetric_regime_v1",
    "swing_trend_simple_candidate_v1",
    "swing_trend_simple_candidate_v2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run confirmed-swing ablation matrix on BTC swing profiles.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default="2024-03-19")
    parser.add_argument("--end", default="2026-03-19")
    parser.add_argument(
        "--strategy-profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated strategy profiles.",
    )
    parser.add_argument(
        "--long-exit-json",
        default='{"take_profit_mode":"scaled","scaled_tp1_r":1.0,"scaled_tp2_r":3.0}',
    )
    parser.add_argument(
        "--short-exit-json",
        default='{"take_profit_mode":"fixed_r","fixed_take_profit_r":1.5}',
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/backtests/btc_confirmed_swing_ablation",
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
            exit_profile="confirmed_swing_ablation_long_scaled1_3_short_fixed1_5",
            take_profit_mode="scaled",
            scaled_tp1_r=1.0,
            scaled_tp2_r=3.0,
            long_exit=parse_exit_json(args.long_exit_json),
            short_exit=parse_exit_json(args.short_exit_json),
            swing_detection_mode="confirmed",
        ),
    )


def format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row[key]) for key, _ in columns) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def main() -> None:
    args = parse_args()
    configure_logging()

    start = parse_date(args.start)
    end = parse_date(args.end)
    profiles = [item.strip() for item in args.strategy_profiles.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    service = build_service(args)

    enriched_history: dict[str, dict[str, pd.DataFrame]] = {}
    for profile in profiles:
        print(f"[confirmed-ablation] preload {profile}", flush=True)
        enriched_history[profile] = service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=profile,
            start=start,
            end=end,
        )

    baseline_profile = profiles[0]
    records: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for profile in profiles:
        print(f"[confirmed-ablation] run {profile}", flush=True)
        summary, trades = service.run_symbol_strategy_with_enriched_frames(
            symbol=args.symbol,
            strategy_profile=profile,
            start=start,
            end=end,
            enriched_frames=enriched_history[profile],
        )
        trades_df = pd.DataFrame([asdict(item) for item in trades])
        if trades_df.empty:
            long_trades = pd.DataFrame()
            short_trades = pd.DataFrame()
        else:
            long_trades = trades_df[trades_df["side"] == "LONG"].copy()
            short_trades = trades_df[trades_df["side"] == "SHORT"].copy()
            for item in trades_df.to_dict(orient="records"):
                item["profile"] = profile
                trade_rows.append(item)

        records.append(
            {
                "profile": profile,
                "total_trades": int(summary.total_trades),
                "win_rate": float(summary.win_rate),
                "profit_factor": float(summary.profit_factor),
                "expectancy_r": float(summary.expectancy_r),
                "cumulative_r": float(summary.cumulative_r),
                "max_drawdown_r": float(summary.max_drawdown_r),
                "avg_holding_bars": float(summary.avg_holding_bars),
                "tp1_hit_rate": float(summary.tp1_hit_rate),
                "tp2_hit_rate": float(summary.tp2_hit_rate),
                "long_trades": int(len(long_trades)),
                "long_r": float(long_trades["pnl_r"].sum()) if not long_trades.empty else 0.0,
                "long_avg_r": float(long_trades["pnl_r"].mean()) if not long_trades.empty else 0.0,
                "short_trades": int(len(short_trades)),
                "short_r": float(short_trades["pnl_r"].sum()) if not short_trades.empty else 0.0,
                "short_avg_r": float(short_trades["pnl_r"].mean()) if not short_trades.empty else 0.0,
            }
        )

    results_df = pd.DataFrame(records)
    baseline_row = results_df.loc[results_df["profile"] == baseline_profile].iloc[0]
    results_df["delta_expectancy_r_vs_baseline"] = results_df["expectancy_r"] - float(baseline_row["expectancy_r"])
    results_df["delta_cumulative_r_vs_baseline"] = results_df["cumulative_r"] - float(baseline_row["cumulative_r"])
    results_df["delta_profit_factor_vs_baseline"] = results_df["profit_factor"] - float(baseline_row["profit_factor"])
    ranked = results_df.sort_values(
        ["expectancy_r", "cumulative_r", "profit_factor", "max_drawdown_r"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    json_path = output_dir / f"confirmed_swing_ablation_{timestamp}.json"
    csv_path = output_dir / f"confirmed_swing_ablation_{timestamp}.csv"
    trades_csv_path = output_dir / f"confirmed_swing_ablation_{timestamp}_trades.csv"
    ranked.to_json(json_path, orient="records", force_ascii=False, indent=2)
    ranked.to_csv(csv_path, index=False)
    pd.DataFrame(trade_rows).to_csv(trades_csv_path, index=False)

    display_rows: list[dict[str, Any]] = []
    for item in ranked.to_dict(orient="records"):
        display_rows.append(
            {
                "profile": item["profile"],
                "trades": int(item["total_trades"]),
                "pf": f'{float(item["profit_factor"]):.2f}',
                "exp": f'{float(item["expectancy_r"]):.3f}',
                "cum": f'{float(item["cumulative_r"]):.2f}',
                "dd": f'{float(item["max_drawdown_r"]):.2f}',
                "long_r": f'{float(item["long_r"]):.2f}',
                "short_r": f'{float(item["short_r"]):.2f}',
                "delta_exp": f'{float(item["delta_expectancy_r_vs_baseline"]):+.3f}',
                "delta_cum": f'{float(item["delta_cumulative_r_vs_baseline"]):+.2f}',
            }
        )

    best = ranked.iloc[0].to_dict()
    markdown_path = output_dir / f"confirmed_swing_ablation_{timestamp}.md"
    markdown = f"""# Confirmed Swing Ablation

生成时间：{datetime.now(timezone.utc).isoformat()}

## 范围

- 标的：`{args.symbol}`
- 窗口：`{args.start} -> {args.end}`
- Swing 模式：`confirmed`
- Exit：`LONG scaled 1R -> 3R, TP1 后保本；SHORT fixed 1.5R`
- 基线：`{baseline_profile}`

这里要明确一点：这份矩阵不是为了再找一版更复杂的策略，而是为了验证“在 no-lookahead 口径下，之前删掉的条件是否还值得删”。

## 排名

{format_table(
    display_rows,
    [
        ("profile", "Profile"),
        ("trades", "交易数"),
        ("pf", "PF"),
        ("exp", "Expectancy"),
        ("cum", "累计R"),
        ("dd", "回撤R"),
        ("long_r", "Long R"),
        ("short_r", "Short R"),
        ("delta_exp", "对基线 Exp"),
        ("delta_cum", "对基线 累计R"),
    ],
)}

## 最优结果

- 最优 profile：`{best["profile"]}`
- Profit Factor：`{float(best["profit_factor"]):.4f}`
- Expectancy：`{float(best["expectancy_r"]):.4f}R`
- 累计：`{float(best["cumulative_r"]):.4f}R`
- 最大回撤：`{float(best["max_drawdown_r"]):.4f}R`
- LONG：`{float(best["long_r"]):.4f}R`
- SHORT：`{float(best["short_r"]):.4f}R`

## 解释边界

- 这份结果已经避开 centered swing 的未来函数问题，但仍然是同一总窗口上的 in-sample ablation。
- 如果某个“更简单”的版本在这里明显退化，就不能再用 centered swing 下的旧结论替它背书。
- 如果某个版本只靠单边贡献转正，后续还需要结合 walk-forward 再看，不应直接升格成主线。

## 原始文件

- JSON：`{json_path}`
- CSV：`{csv_path}`
- Trades CSV：`{trades_csv_path}`
"""
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved markdown: {markdown_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved trades CSV: {trades_csv_path}")


if __name__ == "__main__":
    main()
