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
        "exit_profile": "walk_forward_long_scaled1_3_short_fixed1_5",
        "take_profit_mode": "scaled",
        "scaled_tp1_r": 1.0,
        "scaled_tp2_r": 3.0,
        "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    },
}

RISK_BUDGET_PRESETS: dict[str, dict[str, Any]] = {
    "flat_1_0": {
        "label": "Flat 1.0x",
        "description": "所有交易统一 1.0R 风险，不做 rolling budget。",
        "type": "flat",
        "base": 1.0,
    },
    "loss_streak2_next1_base1_1_cut0_6": {
        "label": "Base 1.1x / 2 losses -> next 0.6x",
        "description": "默认 1.1R；若连续 2 笔亏损，下一笔临时降到 0.6R。",
        "type": "loss_streak_cap",
        "base": 1.1,
        "loss_streak_trigger": 2,
        "cut_size": 0.6,
        "hold_trades": 1,
    },
    "loss_streak2_next1_base1_1_cut0_75": {
        "label": "Base 1.1x / 2 losses -> next 0.75x",
        "description": "默认 1.1R；若连续 2 笔亏损，下一笔临时降到 0.75R。",
        "type": "loss_streak_cap",
        "base": 1.1,
        "loss_streak_trigger": 2,
        "cut_size": 0.75,
        "hold_trades": 1,
    },
    "loss_streak2_next1_base1_15_cut0_6": {
        "label": "Base 1.15x / 2 losses -> next 0.6x",
        "description": "默认 1.15R；若连续 2 笔亏损，下一笔临时降到 0.6R。",
        "type": "loss_streak_cap",
        "base": 1.15,
        "loss_streak_trigger": 2,
        "cut_size": 0.6,
        "hold_trades": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC-only walk-forward / OOS validation for rolling risk budgets.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--strategy-profile", default="swing_trend_long_regime_gate_v1")
    parser.add_argument("--exit-preset", choices=sorted(EXIT_PRESETS.keys()), default="long_scaled1_3_short_fixed1_5")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default="2024-03-19")
    parser.add_argument("--end", default="2026-03-19")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--scheme", choices=["rolling", "anchored"], default="rolling")
    parser.add_argument(
        "--selection-metric",
        choices=["cumulative_r", "return_over_drawdown", "expectancy_r", "profit_factor"],
        default="cumulative_r",
    )
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--budget-min-size-utilization", type=float, default=0.95)
    parser.add_argument("--budget-max-size-utilization", type=float, default=1.05)
    parser.add_argument(
        "--risk-budget-presets",
        default=",".join(RISK_BUDGET_PRESETS.keys()),
        help="Comma-separated risk budget preset keys.",
    )
    parser.add_argument("--output-dir", default="artifacts/backtests/btc_risk_budget_walk_forward")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


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


def parse_presets(raw: str) -> list[str]:
    presets = [item.strip() for item in raw.split(",") if item.strip()]
    if not presets:
        raise ValueError("At least one risk budget preset is required")
    unknown = [item for item in presets if item not in RISK_BUDGET_PRESETS]
    if unknown:
        raise ValueError(f"Unsupported risk budget presets: {', '.join(unknown)}")
    return presets


def apply_risk_budget(trades: list[BacktestTrade], preset: str) -> list[dict[str, Any]]:
    ordered = sorted(trades, key=lambda item: pd.Timestamp(item.entry_time))
    cfg = RISK_BUDGET_PRESETS[preset]
    budget_type = str(cfg["type"])
    base = float(cfg["base"])
    loss_streak_trigger = int(cfg.get("loss_streak_trigger", 0))
    cut_size = float(cfg.get("cut_size", base))
    hold_trades = int(cfg.get("hold_trades", 0))

    rows: list[dict[str, Any]] = []
    loss_streak = 0
    hold_remaining = 0

    for trade in ordered:
        size = base
        if budget_type == "loss_streak_cap" and hold_remaining > 0:
            size = min(size, cut_size)

        weighted_pnl_r = float(trade.pnl_r) * size
        row = {
            **asdict(trade),
            "risk_budget_preset": preset,
            "size_multiplier": round(size, 6),
            "weighted_pnl_r": round(weighted_pnl_r, 6),
        }
        rows.append(row)

        if weighted_pnl_r < 0:
            loss_streak += 1
        else:
            loss_streak = 0

        if budget_type == "loss_streak_cap":
            if loss_streak >= loss_streak_trigger:
                hold_remaining = hold_trades
            elif hold_remaining > 0:
                hold_remaining -= 1

    return rows


def summarize_weighted_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total_trades": 0,
            "avg_size": 0.0,
            "downsized_trades": 0,
            "upsized_trades": 0,
            "size_utilization": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "cumulative_r": 0.0,
            "max_drawdown_r": 0.0,
            "return_over_drawdown": 0.0,
        }

    ordered = sorted(rows, key=lambda item: pd.Timestamp(item["entry_time"]))
    total_trades = len(ordered)
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    positive = 0.0
    negative = 0.0

    for row in ordered:
        weighted_pnl_r = float(row["weighted_pnl_r"])
        cumulative += weighted_pnl_r
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
        if weighted_pnl_r > 0:
            positive += weighted_pnl_r
        elif weighted_pnl_r < 0:
            negative += abs(weighted_pnl_r)

    avg_size = sum(float(row["size_multiplier"]) for row in ordered) / total_trades
    expectancy = sum(float(row["weighted_pnl_r"]) for row in ordered) / total_trades
    profit_factor = (positive / negative) if negative else (999.0 if positive > 0 else 0.0)
    return_over_drawdown = (cumulative / max_drawdown) if max_drawdown > 0 else (999.0 if cumulative > 0 else 0.0)

    return {
        "total_trades": total_trades,
        "avg_size": round(avg_size, 4),
        "downsized_trades": int(sum(1 for row in ordered if float(row["size_multiplier"]) < 1.0)),
        "upsized_trades": int(sum(1 for row in ordered if float(row["size_multiplier"]) > 1.0)),
        "size_utilization": round(avg_size, 4),
        "win_rate": round((sum(1 for row in ordered if float(row["weighted_pnl_r"]) > 0) / total_trades) * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "expectancy_r": round(expectancy, 4),
        "cumulative_r": round(cumulative, 4),
        "max_drawdown_r": round(max_drawdown, 4),
        "return_over_drawdown": round(return_over_drawdown, 4),
    }


def selection_value(summary: dict[str, Any], metric: str) -> float:
    value = float(summary[metric])
    if metric in {"profit_factor", "return_over_drawdown"} and value == 0.0 and float(summary["cumulative_r"]) > 0:
        return 999.0
    return value


def format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging()

    start = parse_date(args.start)
    end = parse_date(args.end)
    presets = parse_presets(args.risk_budget_presets)
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
    print(f"[risk-budget-wf] preload {args.strategy_profile}", flush=True)
    enriched_history = service.prepare_enriched_history(
        exchange=args.exchange,
        market_type=args.market_type,
        symbol=args.symbol,
        strategy_profile=args.strategy_profile,
        start=start,
        end=end,
    )
    print(f"[risk-budget-wf] ready {args.strategy_profile}", flush=True)

    fold_rows: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    selected_oos_rows: list[dict[str, Any]] = []
    flat_oos_rows: list[dict[str, Any]] = []

    for fold in folds:
        print(
            f"[risk-budget-wf] fold {fold['fold']}/{len(folds)} train {fold['train_start'].date()}->{fold['train_end'].date()} test {fold['test_start'].date()}->{fold['test_end'].date()}",
            flush=True,
        )
        _, train_trades = service.run_symbol_strategy_with_enriched_frames(
            symbol=args.symbol,
            strategy_profile=args.strategy_profile,
            start=fold["train_start"],
            end=fold["train_end"],
            enriched_frames=enriched_history,
        )
        _, oos_trades = service.run_symbol_strategy_with_enriched_frames(
            symbol=args.symbol,
            strategy_profile=args.strategy_profile,
            start=fold["test_start"],
            end=fold["test_end"],
            enriched_frames=enriched_history,
        )

        train_candidates: list[dict[str, Any]] = []
        for preset in presets:
            train_rows = apply_risk_budget(train_trades, preset)
            summary = summarize_weighted_rows(train_rows)
            train_candidates.append({"preset": preset, "train": summary})
            leaderboard_rows.append(
                {
                    "fold": fold["fold"],
                    "train_start": fold["train_start"].date().isoformat(),
                    "train_end": fold["train_end"].date().isoformat(),
                    "preset": preset,
                    "label": RISK_BUDGET_PRESETS[preset]["label"],
                    **{f"train_{key}": value for key, value in summary.items()},
                }
            )

        eligible = [
            item
            for item in train_candidates
            if item["train"]["total_trades"] >= args.min_train_trades
            and float(item["train"]["cumulative_r"]) > 0
            and args.budget_min_size_utilization <= float(item["train"]["avg_size"]) <= args.budget_max_size_utilization
        ]
        ranked = sorted(
            eligible,
            key=lambda item: (
                selection_value(item["train"], args.selection_metric),
                float(item["train"]["cumulative_r"]),
                float(item["train"]["profit_factor"]),
            ),
            reverse=True,
        )
        chosen = ranked[0] if ranked else None

        flat_oos = apply_risk_budget(oos_trades, "flat_1_0")
        flat_summary = summarize_weighted_rows(flat_oos)
        flat_oos_rows.extend({**row, "fold": fold["fold"]} for row in flat_oos)

        if chosen is None:
            print(f"[risk-budget-wf] fold {fold['fold']} skipped: no eligible preset", flush=True)
            fold_rows.append(
                {
                    "fold": fold["fold"],
                    "train_start": fold["train_start"].date().isoformat(),
                    "train_end": fold["train_end"].date().isoformat(),
                    "test_start": fold["test_start"].date().isoformat(),
                    "test_end": fold["test_end"].date().isoformat(),
                    "selected_preset": None,
                    "selection_metric": args.selection_metric,
                    "train_metric": None,
                    "oos_selected": None,
                    "oos_flat": flat_summary,
                }
            )
            continue

        print(f"[risk-budget-wf] fold {fold['fold']} selected {chosen['preset']}", flush=True)
        selected_rows = apply_risk_budget(oos_trades, chosen["preset"])
        selected_summary = summarize_weighted_rows(selected_rows)
        selected_oos_rows.extend({**row, "fold": fold["fold"]} for row in selected_rows)
        fold_rows.append(
            {
                "fold": fold["fold"],
                "train_start": fold["train_start"].date().isoformat(),
                "train_end": fold["train_end"].date().isoformat(),
                "test_start": fold["test_start"].date().isoformat(),
                "test_end": fold["test_end"].date().isoformat(),
                "selected_preset": chosen["preset"],
                "selection_metric": args.selection_metric,
                "train_metric": selection_value(chosen["train"], args.selection_metric),
                "oos_selected": selected_summary,
                "oos_flat": flat_summary,
            }
        )

    selected_folds = [row for row in fold_rows if row["selected_preset"]]
    selection_counts: dict[str, int] = {}
    for row in selected_folds:
        selection_counts[row["selected_preset"]] = selection_counts.get(row["selected_preset"], 0) + 1

    aggregate_selected = summarize_weighted_rows(selected_oos_rows)
    aggregate_flat = summarize_weighted_rows(flat_oos_rows)

    fold_json_path = output_dir / f"risk_budget_walk_forward_folds_{timestamp}.json"
    fold_json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": args.symbol,
                "strategy_profile": args.strategy_profile,
                "exit_preset": args.exit_preset,
                "exchange": args.exchange,
                "market_type": args.market_type,
                "overall_start": start.isoformat(),
                "overall_end": end.isoformat(),
                "scheme": args.scheme,
                "train_days": args.train_days,
                "test_days": args.test_days,
                "step_days": args.step_days,
                "selection_metric": args.selection_metric,
                "min_train_trades": args.min_train_trades,
                "budget_min_size_utilization": args.budget_min_size_utilization,
                "budget_max_size_utilization": args.budget_max_size_utilization,
                "swing_detection_mode": "confirmed",
                "risk_budget_presets": [{"key": key, **RISK_BUDGET_PRESETS[key]} for key in presets],
                "folds": fold_rows,
                "selection_counts": selection_counts,
                "aggregate_oos_selected": aggregate_selected,
                "aggregate_oos_flat": aggregate_flat,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    leaderboard_csv = output_dir / f"risk_budget_walk_forward_train_leaderboard_{timestamp}.csv"
    pd.DataFrame(leaderboard_rows).to_csv(leaderboard_csv, index=False)

    selected_oos_csv = output_dir / f"risk_budget_walk_forward_oos_selected_{timestamp}.csv"
    pd.DataFrame(selected_oos_rows).to_csv(selected_oos_csv, index=False)

    flat_oos_csv = output_dir / f"risk_budget_walk_forward_oos_flat_{timestamp}.csv"
    pd.DataFrame(flat_oos_rows).to_csv(flat_oos_csv, index=False)

    fold_display_rows = []
    for row in fold_rows:
        selected = row["oos_selected"] or {}
        flat = row["oos_flat"] or {}
        fold_display_rows.append(
            {
                "fold": row["fold"],
                "train": f'{row["train_start"]} -> {row["train_end"]}',
                "test": f'{row["test_start"]} -> {row["test_end"]}',
                "selected": row["selected_preset"] or "skip",
                "train_metric": "n/a" if row["train_metric"] is None else f'{float(row["train_metric"]):.3f}',
                "oos_sel_cum": "n/a" if not selected else f'{float(selected["cumulative_r"]):.2f}',
                "oos_sel_dd": "n/a" if not selected else f'{float(selected["max_drawdown_r"]):.2f}',
                "oos_flat_cum": "n/a" if not flat else f'{float(flat["cumulative_r"]):.2f}',
                "oos_flat_dd": "n/a" if not flat else f'{float(flat["max_drawdown_r"]):.2f}',
            }
        )

    selection_rows = [
        {"preset": key, "label": RISK_BUDGET_PRESETS[key]["label"], "selected_count": count}
        for key, count in sorted(selection_counts.items(), key=lambda item: item[1], reverse=True)
    ]
    preset_rows = [
        {"preset": key, "label": RISK_BUDGET_PRESETS[key]["label"], "description": RISK_BUDGET_PRESETS[key]["description"]}
        for key in presets
    ]

    markdown = "\n".join(
        [
            "# BTC Rolling Risk Budget Walk-Forward / OOS",
            "",
            f"生成时间：{datetime.now(timezone.utc).isoformat()}",
            "",
            "## 设定",
            "",
            f"- 标的：`{args.symbol}`",
            f"- 策略：`{args.strategy_profile}`",
            f"- Exit preset：`{args.exit_preset}`",
            f"- 总窗口：`{args.start} -> {args.end}`",
            f"- 训练窗：`{args.train_days}` 天",
            f"- 测试窗：`{args.test_days}` 天",
            f"- 步长：`{args.step_days}` 天",
            f"- 方案：`{args.scheme}`",
            f"- 训练选择指标：`{args.selection_metric}`",
            f"- 预算带：`{args.budget_min_size_utilization:.2f}x -> {args.budget_max_size_utilization:.2f}x`",
            f"- Backtest swing 模式：`confirmed`",
            "",
            "这里不是传统多标的组合风控，而是单策略单标的下的 rolling trade budget。",
            "",
            "## 候选",
            "",
            format_table(preset_rows, [("preset", "候选"), ("label", "标签"), ("description", "说明")]),
            "",
            "## Fold 结果",
            "",
            format_table(
                fold_display_rows,
                [
                    ("fold", "Fold"),
                    ("train", "训练窗"),
                    ("test", "测试窗"),
                    ("selected", "选中预算"),
                    ("train_metric", "训练指标"),
                    ("oos_sel_cum", "OOS 选中累计R"),
                    ("oos_sel_dd", "OOS 选中回撤R"),
                    ("oos_flat_cum", "OOS Flat累计R"),
                    ("oos_flat_dd", "OOS Flat回撤R"),
                ],
            ),
            "",
            "## 选择频次",
            "",
            format_table(selection_rows, [("preset", "候选"), ("label", "标签"), ("selected_count", "被选次数")]) if selection_rows else "无",
            "",
            "## OOS 汇总",
            "",
            f"- 选中预算 OOS 累计：`{float(aggregate_selected['cumulative_r']):.4f}R`",
            f"- 选中预算 OOS 最大回撤：`{float(aggregate_selected['max_drawdown_r']):.4f}R`",
            f"- 选中预算 OOS Profit Factor：`{float(aggregate_selected['profit_factor']):.4f}`",
            f"- 选中预算 OOS Expectancy：`{float(aggregate_selected['expectancy_r']):.4f}R`",
            f"- 选中预算 OOS 平均 size：`{float(aggregate_selected['avg_size']):.4f}`",
            f"- Flat 基线 OOS 累计：`{float(aggregate_flat['cumulative_r']):.4f}R`",
            f"- Flat 基线 OOS 最大回撤：`{float(aggregate_flat['max_drawdown_r']):.4f}R`",
            f"- Flat 基线 OOS Profit Factor：`{float(aggregate_flat['profit_factor']):.4f}`",
            f"- Flat 基线 OOS Expectancy：`{float(aggregate_flat['expectancy_r']):.4f}R`",
            "",
            "## 原始文件",
            "",
            f"- Fold JSON：`{fold_json_path}`",
            f"- 训练榜单 CSV：`{leaderboard_csv}`",
            f"- OOS 选中预算 trades CSV：`{selected_oos_csv}`",
            f"- OOS Flat trades CSV：`{flat_oos_csv}`",
        ]
    )
    markdown_path = output_dir / f"risk_budget_walk_forward_report_{timestamp}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved markdown: {markdown_path}")
    print(f"Saved fold JSON: {fold_json_path}")
    print(f"Saved train leaderboard CSV: {leaderboard_csv}")
    print(f"Saved OOS selected trades CSV: {selected_oos_csv}")
    print(f"Saved OOS flat trades CSV: {flat_oos_csv}")


if __name__ == "__main__":
    main()
