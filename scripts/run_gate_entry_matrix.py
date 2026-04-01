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


GROUPS = (
    {
        "id": "1",
        "label": "gate + simple entry",
        "profile": "swing_trend_matrix_gate_simple_entry_v1",
        "description": "保留当前 regime admission，只用趋势内固定回踩执行。",
    },
    {
        "id": "2",
        "label": "no gate + current entry",
        "profile": "swing_trend_matrix_no_gate_current_entry_v1",
        "description": "移除 trend-strength admission，保留当前 reversal + regained_fast + held_slow + auxiliary entry。",
    },
    {
        "id": "3",
        "label": "gate + current entry",
        "profile": "swing_trend_long_regime_gate_v1",
        "description": "当前主线，对照组。",
    },
    {
        "id": "4",
        "label": "no gate + simple entry",
        "profile": "swing_trend_matrix_no_gate_simple_entry_v1",
        "description": "去掉 admission，也去掉当前 entry 细节，只保留趋势内固定回踩执行。",
    },
)

EXIT_ASSUMPTIONS = {
    "exit_profile": "gate_entry_matrix_long_scaled1_3_short_fixed1_5",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}


@dataclass(frozen=True)
class FoldWindow:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 2x2 gate/entry matrix backtest and walk-forward evaluation.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default="2024-03-19")
    parser.add_argument("--end", default="2026-03-19")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--scheme", choices=["rolling", "anchored"], default="rolling")
    parser.add_argument("--output-dir", default="artifacts/backtests/gate_entry_matrix")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_service() -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**EXIT_ASSUMPTIONS),
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


def summary_to_dict(summary) -> dict[str, Any]:
    return {
        "total_trades": int(summary.total_trades),
        "win_rate": float(summary.win_rate),
        "profit_factor": float(summary.profit_factor),
        "expectancy_r": float(summary.expectancy_r),
        "cumulative_r": float(summary.cumulative_r),
        "max_drawdown_r": float(summary.max_drawdown_r),
        "signals_now": int(summary.signals_now),
        "skipped_entries": int(summary.skipped_entries),
    }


def fmt_metric(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row[key]) for key, _ in columns) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def build_full_window_rows(report) -> list[dict[str, Any]]:
    by_profile = {item.strategy_profile: item for item in report.overall}
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        summary = by_profile[group["profile"]]
        rows.append(
            {
                "group": group["id"],
                "label": group["label"],
                "profile": group["profile"],
                "trades": summary.total_trades,
                "pf": fmt_metric(summary.profit_factor, 4),
                "exp_r": fmt_metric(summary.expectancy_r, 4),
                "cum_r": fmt_metric(summary.cumulative_r, 4),
                "dd_r": fmt_metric(summary.max_drawdown_r, 4),
            }
        )
    return rows


def run_walk_forward_matrix(
    *,
    service: BacktestService,
    symbol: str,
    exchange: str,
    market_type: str,
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    scheme: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    folds = generate_folds(
        start=start,
        end=end,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        scheme=scheme,
    )
    profile_frames = {
        group["profile"]: service.prepare_history(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            strategy_profile=group["profile"],
            start=start,
            end=end,
        )
        for group in GROUPS
    }

    fold_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for group in GROUPS:
        profile = group["profile"]
        frames = profile_frames[profile]
        oos_trades: list[BacktestTrade] = []
        oos_signals_now = 0
        oos_skipped_entries = 0

        for fold in folds:
            train_summary, _ = service.run_symbol_strategy_with_frames(
                symbol=symbol,
                strategy_profile=profile,
                start=fold.train_start,
                end=fold.train_end,
                frames=frames,
            )
            test_summary, test_trades = service.run_symbol_strategy_with_frames(
                symbol=symbol,
                strategy_profile=profile,
                start=fold.test_start,
                end=fold.test_end,
                frames=frames,
            )
            oos_trades.extend(test_trades)
            oos_signals_now += test_summary.signals_now
            oos_skipped_entries += test_summary.skipped_entries

            fold_rows.append(
                {
                    "group": group["id"],
                    "label": group["label"],
                    "profile": profile,
                    "fold": fold.index,
                    "train_start": fold.train_start.date().isoformat(),
                    "train_end": fold.train_end.date().isoformat(),
                    "test_start": fold.test_start.date().isoformat(),
                    "test_end": fold.test_end.date().isoformat(),
                    "train_trades": train_summary.total_trades,
                    "train_pf": round(train_summary.profit_factor, 4),
                    "train_exp_r": round(train_summary.expectancy_r, 4),
                    "train_cum_r": round(train_summary.cumulative_r, 4),
                    "test_trades": test_summary.total_trades,
                    "test_pf": round(test_summary.profit_factor, 4),
                    "test_exp_r": round(test_summary.expectancy_r, 4),
                    "test_cum_r": round(test_summary.cumulative_r, 4),
                    "test_dd_r": round(test_summary.max_drawdown_r, 4),
                }
            )

        aggregate_summary = service._summarize_trades(
            trades=oos_trades,
            strategy_profile=profile,
            symbol=symbol,
            signals_now=oos_signals_now,
            skipped_entries=oos_skipped_entries,
        )
        aggregate_rows.append(
            {
                "group": group["id"],
                "label": group["label"],
                "profile": profile,
                "folds": len(folds),
                "trades": aggregate_summary.total_trades,
                "pf": fmt_metric(aggregate_summary.profit_factor, 4),
                "exp_r": fmt_metric(aggregate_summary.expectancy_r, 4),
                "cum_r": fmt_metric(aggregate_summary.cumulative_r, 4),
                "dd_r": fmt_metric(aggregate_summary.max_drawdown_r, 4),
                "signals_now": aggregate_summary.signals_now,
                "skipped_entries": aggregate_summary.skipped_entries,
            }
        )
        trade_rows.extend(asdict(item) for item in oos_trades)

    return fold_rows, aggregate_rows, trade_rows


def build_interpretation(full_rows: list[dict[str, Any]], oos_rows: list[dict[str, Any]]) -> list[str]:
    full_by_group = {row["group"]: row for row in full_rows}
    oos_by_group = {row["group"]: row for row in oos_rows}

    def parse_float(rows: dict[str, dict[str, Any]], group: str, key: str) -> float:
        return float(rows[group][key])

    notes: list[str] = []

    full_gap_13 = parse_float(full_by_group, "3", "cum_r") - parse_float(full_by_group, "1", "cum_r")
    oos_gap_13 = parse_float(oos_by_group, "3", "cum_r") - parse_float(oos_by_group, "1", "cum_r")
    if abs(oos_gap_13) <= 2.0:
        notes.append("1) 和 3) 在 OOS 很接近，说明 admission 价值大于当前 entry 细节。")
    else:
        notes.append(f"3) 比 1) 的 OOS 累计多 {oos_gap_13:.2f}R，说明当前 entry 细节仍有保留价值。")

    if parse_float(oos_by_group, "2", "cum_r") < 0:
        notes.append("2) 在 OOS 为负，说明移除 gate 后，当前 entry 细节本身不足以独立支撑策略。")
    else:
        notes.append("2) 在 OOS 没有直接失真，说明 gate 不是唯一支撑，entry 自身仍有一定贡献。")

    full_gap_24 = parse_float(full_by_group, "2", "cum_r") - parse_float(full_by_group, "4", "cum_r")
    oos_gap_24 = parse_float(oos_by_group, "2", "cum_r") - parse_float(oos_by_group, "4", "cum_r")
    if oos_gap_24 > 2.0:
        notes.append(f"2) 比 4) 的 OOS 累计多 {oos_gap_24:.2f}R，说明 current entry 即使在无 gate 条件下也优于 simple entry。")
    else:
        notes.append("2) 和 4) 的 OOS 差距不大，说明 current entry 细节在无 gate 条件下贡献有限。")

    if full_gap_13 > 0 and oos_gap_13 > 0:
        notes.append("3) 同时在全窗口和 OOS 都优于 1)，这更像 entry 细节确实有增益，不只是 in-sample 偏差。")

    return notes


def write_outputs(
    *,
    output_dir: Path,
    full_report,
    full_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
    oos_trades: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    BacktestService.save_report(full_report, output_dir)

    pd.DataFrame(fold_rows).to_csv(output_dir / "gate_entry_matrix_folds.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(output_dir / "gate_entry_matrix_oos_summary.csv", index=False)
    pd.DataFrame(oos_trades).to_csv(output_dir / "gate_entry_matrix_oos_trades.csv", index=False)
    (output_dir / "gate_entry_matrix_results.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": args.symbol,
                "exchange": args.exchange,
                "market_type": args.market_type,
                "start": args.start,
                "end": args.end,
                "train_days": args.train_days,
                "test_days": args.test_days,
                "step_days": args.step_days,
                "scheme": args.scheme,
                "groups": list(GROUPS),
                "full_window": full_rows,
                "walk_forward_folds": fold_rows,
                "walk_forward_oos": oos_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    definition_rows = [{"group": item["id"], "label": item["label"], "profile": item["profile"], "description": item["description"]} for item in GROUPS]
    interpretation = build_interpretation(full_rows, oos_rows)
    markdown = "\n".join(
        [
            "# Gate / Entry Matrix",
            "",
            f"生成时间：{datetime.now(timezone.utc).isoformat()}",
            "",
            "## 设定",
            "",
            f"- 标的：`{args.symbol}`",
            f"- 总窗口：`{args.start} -> {args.end}`",
            f"- Walk-forward：`{args.train_days}d train / {args.test_days}d test / {args.step_days}d step / {args.scheme}`",
            "- Exit：`LONG scaled 1R -> 3R / SHORT fixed 1.5R`",
            "- Swing 模式：`confirmed`",
            "",
            "## 四组定义",
            "",
            render_table(
                definition_rows,
                [
                    ("group", "组"),
                    ("label", "标签"),
                    ("profile", "Profile"),
                    ("description", "定义"),
                ],
            ),
            "",
            "## 全窗口结果",
            "",
            render_table(
                full_rows,
                [
                    ("group", "组"),
                    ("label", "标签"),
                    ("trades", "交易数"),
                    ("pf", "PF"),
                    ("exp_r", "Exp"),
                    ("cum_r", "累计R"),
                    ("dd_r", "回撤R"),
                ],
            ),
            "",
            "## Walk-Forward / OOS 汇总",
            "",
            render_table(
                oos_rows,
                [
                    ("group", "组"),
                    ("label", "标签"),
                    ("trades", "OOS交易数"),
                    ("pf", "OOS PF"),
                    ("exp_r", "OOS Exp"),
                    ("cum_r", "OOS累计R"),
                    ("dd_r", "OOS回撤R"),
                ],
            ),
            "",
            "## 解读",
            "",
            *[f"- {item}" for item in interpretation],
            "",
            "## 原始文件",
            "",
            "- 全窗口 JSON / CSV：同目录标准 backtest 导出",
            "- Fold CSV：`gate_entry_matrix_folds.csv`",
            "- OOS 汇总 CSV：`gate_entry_matrix_oos_summary.csv`",
            "- OOS trades CSV：`gate_entry_matrix_oos_trades.csv`",
            "- 汇总 JSON：`gate_entry_matrix_results.json`",
        ]
    )
    (output_dir / "gate_entry_matrix_report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()

    start = parse_date(args.start)
    end = parse_date(args.end)
    service = build_service()
    profiles = [item["profile"] for item in GROUPS]

    full_report = service.run(
        exchange=args.exchange,
        market_type=args.market_type,
        symbols=[args.symbol],
        strategy_profiles=profiles,
        start=start,
        end=end,
    )
    full_rows = build_full_window_rows(full_report)
    fold_rows, oos_rows, oos_trades = run_walk_forward_matrix(
        service=service,
        symbol=args.symbol,
        exchange=args.exchange,
        market_type=args.market_type,
        start=start,
        end=end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        scheme=args.scheme,
    )
    write_outputs(
        output_dir=Path(args.output_dir),
        full_report=full_report,
        full_rows=full_rows,
        fold_rows=fold_rows,
        oos_rows=oos_rows,
        oos_trades=oos_trades,
        args=args,
    )

    print("Full window:")
    for row in full_rows:
        print(
            f"  {row['group']}. {row['label']}: trades={row['trades']} pf={row['pf']} "
            f"exp={row['exp_r']} cum_r={row['cum_r']} dd_r={row['dd_r']}"
        )
    print("OOS:")
    for row in oos_rows:
        print(
            f"  {row['group']}. {row['label']}: trades={row['trades']} pf={row['pf']} "
            f"exp={row['exp_r']} cum_r={row['cum_r']} dd_r={row['dd_r']}"
        )
    print(f"Saved outputs under: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
