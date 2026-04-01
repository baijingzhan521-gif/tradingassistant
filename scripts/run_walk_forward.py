from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
    "scaled_current": {
        "exit_profile": "walk_forward_scaled_current",
        "take_profit_mode": "scaled",
    },
    "fixed_2r": {
        "exit_profile": "walk_forward_fixed_2r",
        "take_profit_mode": "fixed_r",
        "fixed_take_profit_r": 2.0,
    },
}

DEFAULT_CANDIDATES = (
    "swing_trend_long_regime_gate_v1@long_scaled1_3_short_fixed1_5",
    "swing_trend_ablation_no_reversal_v1@long_scaled1_3_short_fixed1_5",
    "swing_trend_simple_candidate_v1@long_scaled1_3_short_fixed1_5",
    "swing_trend_simple_candidate_v2@long_scaled1_3_short_fixed1_5",
)


@dataclass(frozen=True)
class CandidateSpec:
    strategy_profile: str
    exit_preset: str

    @property
    def key(self) -> str:
        return f"{self.strategy_profile}@{self.exit_preset}"


@dataclass(frozen=True)
class FoldWindow:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC-only walk-forward / OOS validation.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default="2024-03-19")
    parser.add_argument("--end", default="2026-03-19")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--scheme", choices=["rolling", "anchored"], default="rolling")
    parser.add_argument("--selection-metric", choices=["expectancy_r", "cumulative_r", "profit_factor"], default="expectancy_r")
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--allow-nonpositive-train-selection", action="store_true")
    parser.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
        help="Comma-separated candidate specs. Format: strategy_profile@exit_preset",
    )
    parser.add_argument("--output-dir", default="artifacts/backtests/btc_walk_forward")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def parse_candidates(raw: str) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if "@" in item:
            strategy_profile, exit_preset = item.split("@", 1)
        else:
            strategy_profile, exit_preset = item, "long_scaled1_3_short_fixed1_5"
        if exit_preset not in EXIT_PRESETS:
            raise ValueError(f"Unsupported exit preset: {exit_preset}")
        specs.append(CandidateSpec(strategy_profile=strategy_profile, exit_preset=exit_preset))
    if not specs:
        raise ValueError("At least one candidate is required")
    return specs


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


def generate_folds(
    *,
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    scheme: str,
) -> list[FoldWindow]:
    folds: list[FoldWindow] = []
    anchor_start = start
    train_start = start
    train_end = train_start + timedelta(days=train_days)
    index = 1

    while train_end + timedelta(days=test_days) <= end:
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        folds.append(
            FoldWindow(
                index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
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


def selection_value(summary_dict: dict[str, Any], metric: str) -> float:
    value = float(summary_dict[metric])
    if metric == "profit_factor" and value == 0.0 and summary_dict["cumulative_r"] > 0:
        return 999.0
    return value


def passes_positive_gate(summary_dict: dict[str, Any], metric: str) -> bool:
    if metric == "profit_factor":
        return float(summary_dict["profit_factor"]) > 1.0
    return float(summary_dict[metric]) > 0.0


def rank_candidates(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            selection_value(item["train"], metric),
            float(item["train"]["cumulative_r"]),
            float(item["train"]["profit_factor"]),
            float(item["train"]["expectancy_r"]),
            -float(item["train"]["max_drawdown_r"]),
        ),
        reverse=True,
    )


def format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row[key]) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body])


def aggregate_by_side(trades_df: pd.DataFrame) -> list[dict[str, Any]]:
    if trades_df.empty:
        return []
    grouped = trades_df.groupby("side", as_index=False).agg(
        trades=("pnl_r", "size"),
        cumulative_r=("pnl_r", "sum"),
        avg_r=("pnl_r", "mean"),
        win_rate=("pnl_r", lambda s: float((s > 0).mean() * 100)),
    )
    rows: list[dict[str, Any]] = []
    for item in grouped.to_dict(orient="records"):
        rows.append(
            {
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
    candidates = parse_candidates(args.candidates)
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

    service_by_exit = {preset: build_service(preset) for preset in {spec.exit_preset for spec in candidates}}
    history_service = next(iter(service_by_exit.values()))
    enriched_history: dict[str, dict[str, pd.DataFrame]] = {}
    for profile in sorted({spec.strategy_profile for spec in candidates}):
        print(f"[walk-forward] preload {profile}", flush=True)
        enriched_history[profile] = history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=profile,
            start=start,
            end=end,
        )
        print(f"[walk-forward] ready {profile}", flush=True)

    fold_rows: list[dict[str, Any]] = []
    train_leaderboard_rows: list[dict[str, Any]] = []
    oos_trade_rows: list[dict[str, Any]] = []

    for fold in folds:
        print(
            f"[walk-forward] fold {fold.index}/{len(folds)} train {fold.train_start.date()}->{fold.train_end.date()} test {fold.test_start.date()}->{fold.test_end.date()}",
            flush=True,
        )
        train_candidates: list[dict[str, Any]] = []
        for spec in candidates:
            service = service_by_exit[spec.exit_preset]
            train_summary, _ = service.run_symbol_strategy_with_enriched_frames(
                symbol=args.symbol,
                strategy_profile=spec.strategy_profile,
                start=fold.train_start,
                end=fold.train_end,
                enriched_frames=enriched_history[spec.strategy_profile],
            )
            row = {
                "fold": fold.index,
                "candidate": spec.key,
                "strategy_profile": spec.strategy_profile,
                "exit_preset": spec.exit_preset,
                "train": to_summary_dict(train_summary),
            }
            train_candidates.append(row)
            train_leaderboard_rows.append(
                {
                    "fold": fold.index,
                    "train_start": fold.train_start.date().isoformat(),
                    "train_end": fold.train_end.date().isoformat(),
                    "candidate": spec.key,
                    **{f"train_{key}": value for key, value in row["train"].items()},
                }
            )

        eligible = [item for item in train_candidates if item["train"]["total_trades"] >= args.min_train_trades]
        ranked = rank_candidates(eligible, args.selection_metric)
        chosen = ranked[0] if ranked else None
        if chosen and (not args.allow_nonpositive_train_selection) and not passes_positive_gate(chosen["train"], args.selection_metric):
            chosen = None

        if chosen is None:
            print(f"[walk-forward] fold {fold.index} skipped: no eligible positive candidate", flush=True)
            fold_rows.append(
                {
                    "fold": fold.index,
                    "train_start": fold.train_start.date().isoformat(),
                    "train_end": fold.train_end.date().isoformat(),
                    "test_start": fold.test_start.date().isoformat(),
                    "test_end": fold.test_end.date().isoformat(),
                    "selected_candidate": None,
                    "selection_metric": args.selection_metric,
                    "train_metric": None,
                    "train_trades": 0,
                    "oos": None,
                }
            )
            continue

        print(f"[walk-forward] fold {fold.index} selected {chosen['candidate']}", flush=True)
        service = service_by_exit[chosen["exit_preset"]]
        oos_summary, oos_trades = service.run_symbol_strategy_with_enriched_frames(
            symbol=args.symbol,
            strategy_profile=chosen["strategy_profile"],
            start=fold.test_start,
            end=fold.test_end,
            enriched_frames=enriched_history[chosen["strategy_profile"]],
        )
        oos_dict = to_summary_dict(oos_summary)
        fold_rows.append(
            {
                "fold": fold.index,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "test_start": fold.test_start.date().isoformat(),
                "test_end": fold.test_end.date().isoformat(),
                "selected_candidate": chosen["candidate"],
                "selection_metric": args.selection_metric,
                "train_metric": selection_value(chosen["train"], args.selection_metric),
                "train_trades": chosen["train"]["total_trades"],
                "oos": oos_dict,
            }
        )
        for trade in oos_trades:
            item = asdict(trade)
            item["fold"] = fold.index
            item["selected_candidate"] = chosen["candidate"]
            oos_trade_rows.append(item)

    oos_trades_df = pd.DataFrame(oos_trade_rows)
    if not oos_trades_df.empty:
        oos_trades_df["entry_time"] = pd.to_datetime(oos_trades_df["entry_time"], utc=True)

    selected_folds = [row for row in fold_rows if row["selected_candidate"]]
    base_service = history_service
    aggregate_oos_summary = base_service._summarize_trades(
        trades=[],
        strategy_profile="walk_forward_oos",
        symbol=args.symbol,
        signals_now=0,
        skipped_entries=0,
    )
    if oos_trade_rows:
        aggregate_oos_summary = base_service._summarize_trades(
            trades=[BacktestTrade(**{key: row[key] for key in BacktestTrade.__dataclass_fields__.keys()}) for row in oos_trade_rows],
            strategy_profile="walk_forward_oos",
            symbol=args.symbol,
            signals_now=0,
            skipped_entries=0,
        )

    selection_counts: dict[str, int] = {}
    for row in selected_folds:
        selection_counts[row["selected_candidate"]] = selection_counts.get(row["selected_candidate"], 0) + 1

    fold_json_path = output_dir / f"walk_forward_folds_{timestamp}.json"
    fold_json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": args.symbol,
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
                "allow_nonpositive_train_selection": args.allow_nonpositive_train_selection,
                "swing_detection_mode": "confirmed",
                "candidates": [asdict(item) for item in candidates],
                "folds": fold_rows,
                "selection_counts": selection_counts,
                "aggregate_oos": to_summary_dict(aggregate_oos_summary),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    leaderboard_csv = output_dir / f"walk_forward_train_leaderboard_{timestamp}.csv"
    pd.DataFrame(train_leaderboard_rows).to_csv(leaderboard_csv, index=False)

    oos_trades_csv = output_dir / f"walk_forward_oos_trades_{timestamp}.csv"
    if oos_trades_df.empty:
        pd.DataFrame(columns=["fold"]).to_csv(oos_trades_csv, index=False)
    else:
        oos_trades_df.to_csv(oos_trades_csv, index=False)

    fold_display_rows: list[dict[str, Any]] = []
    for row in fold_rows:
        oos = row["oos"] or {}
        fold_display_rows.append(
            {
                "fold": row["fold"],
                "train": f'{row["train_start"]} -> {row["train_end"]}',
                "test": f'{row["test_start"]} -> {row["test_end"]}',
                "selected": row["selected_candidate"] or "skip",
                "train_metric": "n/a" if row["train_metric"] is None else f'{float(row["train_metric"]):.3f}',
                "oos_trades": oos.get("total_trades", 0),
                "oos_pf": "n/a" if not oos else f'{float(oos["profit_factor"]):.2f}',
                "oos_exp": "n/a" if not oos else f'{float(oos["expectancy_r"]):.3f}',
                "oos_cum": "n/a" if not oos else f'{float(oos["cumulative_r"]):.2f}',
            }
        )

    selection_rows = [
        {"candidate": key, "folds": value}
        for key, value in sorted(selection_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    oos_positive_ratio = (
        sum(1 for row in selected_folds if row["oos"] and row["oos"]["cumulative_r"] > 0) / len(selected_folds)
        if selected_folds
        else 0.0
    )
    side_rows = aggregate_by_side(oos_trades_df)

    markdown_path = output_dir / f"walk_forward_report_{timestamp}.md"
    markdown = f"""# BTC Walk-Forward / OOS

生成时间：{datetime.now(timezone.utc).isoformat()}

## 设定

- 标的：`{args.symbol}`
- 总窗口：`{args.start} -> {args.end}`
- 训练窗：`{args.train_days}` 天
- 测试窗：`{args.test_days}` 天
- 步长：`{args.step_days}` 天
- 方案：`{args.scheme}`
- 训练选择指标：`{args.selection_metric}`
- 最低训练交易数：`{args.min_train_trades}`
- 允许选择非正训练候选：`{args.allow_nonpositive_train_selection}`
- Backtest swing 模式：`confirmed`

## 候选池

{format_table(
    [{"candidate": item.key, "strategy": item.strategy_profile, "exit": item.exit_preset} for item in candidates],
    [("candidate", "候选"), ("strategy", "策略"), ("exit", "Exit")],
)}

这里特意没有把 divergence / free-space 这类更复杂的研究分支放回默认候选池，避免把 walk-forward 变成过拟合机器。

## Fold 结果

{format_table(
    fold_display_rows,
    [
        ("fold", "Fold"),
        ("train", "训练窗"),
        ("test", "测试窗"),
        ("selected", "选中候选"),
        ("train_metric", "训练指标"),
        ("oos_trades", "OOS 交易数"),
        ("oos_pf", "OOS PF"),
        ("oos_exp", "OOS Exp"),
        ("oos_cum", "OOS 累计R"),
    ],
)}

## 选择频次

{format_table(selection_rows, [("candidate", "候选"), ("folds", "被选次数")])}

## OOS 汇总

- 有效 OOS fold 数：`{len(selected_folds)} / {len(folds)}`
- OOS 正收益 fold 占比：`{oos_positive_ratio:.2%}`
- OOS 总交易数：`{aggregate_oos_summary.total_trades}`
- OOS Profit Factor：`{aggregate_oos_summary.profit_factor:.4f}`
- OOS Expectancy：`{aggregate_oos_summary.expectancy_r:.4f}R`
- OOS 累计：`{aggregate_oos_summary.cumulative_r:.4f}R`
- OOS 最大回撤：`{aggregate_oos_summary.max_drawdown_r:.4f}R`

## OOS 按 side

{format_table(side_rows, [("side", "方向"), ("trades", "交易数"), ("cum_r", "累计R"), ("avg_r", "平均R"), ("win_rate", "胜率")])}

## 解释边界

- 这是一版真正的 train/OOS 分离框架：每个 fold 先在训练窗选候选，再去后面的测试窗验证。
- 这版还不是“参数搜索框架”，而是“受控候选选择框架”。这是刻意收窄，不是功能缺失。
- 为了避免 pivot 偷看未来，walk-forward 使用的是 `confirmed swing`，所以它的结果不应和旧的 centered swing 回测直接横比。

## 原始文件

- Fold JSON：`{fold_json_path}`
- 训练榜单 CSV：`{leaderboard_csv}`
- OOS trades CSV：`{oos_trades_csv}`
"""
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved markdown: {markdown_path}")
    print(f"Saved fold JSON: {fold_json_path}")
    print(f"Saved train leaderboard CSV: {leaderboard_csv}")
    print(f"Saved OOS trades CSV: {oos_trades_csv}")


if __name__ == "__main__":
    main()
