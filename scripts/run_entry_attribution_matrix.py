from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import (
    BacktestAssumptions,
    BacktestReport,
    BacktestService,
    _PendingEntry,
)
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from app.strategies.swing_trend_entry_attribution import (
    SwingTrendEntryAttributionStrategy,
    build_entry_attribution_config,
)
from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    SwingTrendLongRegimeGateV1Strategy,
)
from app.schemas.common import Action, RecommendedTiming


EXIT_ASSUMPTIONS = {
    "exit_profile": "entry_attribution_long_scaled1_3_short_fixed1_5",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}

COMPONENT_FIELDS = (
    ("reversal", "R"),
    ("regained_fast", "RF"),
    ("held_slow", "HS"),
    ("auxiliary", "AUX"),
)


@dataclass(frozen=True)
class EntryComboSpec:
    include_reversal: bool
    include_regained_fast: bool
    include_held_slow: bool
    include_auxiliary: bool

    @property
    def component_map(self) -> dict[str, bool]:
        return {
            "reversal": self.include_reversal,
            "regained_fast": self.include_regained_fast,
            "held_slow": self.include_held_slow,
            "auxiliary": self.include_auxiliary,
        }

    @property
    def profile_name(self) -> str:
        return (
            "entry_attr_"
            f"r{int(self.include_reversal)}_"
            f"rf{int(self.include_regained_fast)}_"
            f"hs{int(self.include_held_slow)}_"
            f"aux{int(self.include_auxiliary)}"
        )

    @property
    def label(self) -> str:
        return " ".join(f"{short}{int(self.component_map[name])}" for name, short in COMPONENT_FIELDS)

    @property
    def enabled_count(self) -> int:
        return sum(int(value) for value in self.component_map.values())


@dataclass(frozen=True)
class FoldWindow:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current-entry attribution matrix with gate fixed on.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default="2024-03-19")
    parser.add_argument("--end", default="2026-03-19")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--scheme", choices=["rolling", "anchored"], default="rolling")
    parser.add_argument("--output-dir", default="artifacts/backtests/entry_attribution_matrix")
    parser.add_argument(
        "--profiles",
        default=None,
        help="Optional comma-separated subset of profile names, e.g. entry_attr_r1_rf1_hs1_aux1,entry_attr_r1_rf1_hs1_aux0",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_service() -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**EXIT_ASSUMPTIONS),
    )


def generate_specs() -> list[EntryComboSpec]:
    specs = [
        EntryComboSpec(
            include_reversal=bool(reversal),
            include_regained_fast=bool(regained_fast),
            include_held_slow=bool(held_slow),
            include_auxiliary=bool(auxiliary),
        )
        for reversal, regained_fast, held_slow, auxiliary in product((0, 1), repeat=4)
    ]
    return sorted(
        specs,
        key=lambda item: (
            item.enabled_count,
            int(item.include_reversal),
            int(item.include_regained_fast),
            int(item.include_held_slow),
            int(item.include_auxiliary),
        ),
    )


def filter_specs(specs: list[EntryComboSpec], raw_profiles: str | None) -> list[EntryComboSpec]:
    if not raw_profiles:
        return specs
    wanted = {item.strip() for item in raw_profiles.split(",") if item.strip()}
    filtered = [spec for spec in specs if spec.profile_name in wanted]
    missing = sorted(wanted - {spec.profile_name for spec in filtered})
    if missing:
        raise ValueError(f"Unknown profile(s): {', '.join(missing)}")
    if not filtered:
        raise ValueError("At least one profile must be selected")
    return filtered


def build_strategy(spec: EntryComboSpec) -> SwingTrendEntryAttributionStrategy:
    return SwingTrendEntryAttributionStrategy(
        build_entry_attribution_config(
            include_reversal=spec.include_reversal,
            include_regained_fast=spec.include_regained_fast,
            include_held_slow=spec.include_held_slow,
            include_auxiliary=spec.include_auxiliary,
        ),
        profile_name=spec.profile_name,
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


def run_custom_strategy_with_enriched_frames(
    *,
    service: BacktestService,
    strategy: SwingTrendEntryAttributionStrategy,
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
    enriched: dict[str, pd.DataFrame],
):
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    required = tuple(strategy.required_timeframes)
    trigger_frame = enriched[trigger_tf]
    trigger_end_idx = int(trigger_frame["timestamp"].searchsorted(pd.Timestamp(end), side="left"))
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}

    trades = []
    pending_entry = None
    position = None
    signals_now = 0
    skipped_entries = 0
    cooldown_remaining = 0
    cooldown_bars_after_exit = int(strategy.config.get("backtest", {}).get("cooldown_bars_after_exit", 0))

    for trigger_idx in range(trigger_end_idx):
        candle = trigger_frame.iloc[trigger_idx]
        ts = candle["timestamp"].to_pydatetime()

        if pending_entry is not None:
            maybe_position = service._open_pending_entry(
                symbol=symbol,
                strategy_profile=strategy_profile,
                pending_entry=pending_entry,
                candle=candle,
            )
            if maybe_position is None:
                skipped_entries += 1
            else:
                position = maybe_position
            pending_entry = None

        if position is not None:
            trade = service._update_open_position(
                position=position,
                candle=candle,
                max_hold_bars=service._max_hold_bars(strategy_profile),
            )
            if trade is not None:
                trades.append(trade)
                position = None
                cooldown_remaining = cooldown_bars_after_exit

        if ts < start:
            continue

        current_indices: dict[str, int] = {trigger_tf: trigger_idx}
        ready = True
        for timeframe in required:
            if timeframe == trigger_tf:
                continue
            frame = enriched[timeframe]
            pointer = indices[timeframe]
            while pointer + 1 < len(frame) and frame.iloc[pointer + 1]["timestamp"] <= candle["timestamp"]:
                pointer += 1
            indices[timeframe] = pointer
            if frame.iloc[pointer]["timestamp"] > candle["timestamp"]:
                ready = False
                break
            current_indices[timeframe] = pointer
        if not ready:
            continue

        min_required = max(int(service.assumptions.lookback // 3), 20)
        if any(index < min_required for index in current_indices.values()):
            continue

        if position is not None or pending_entry is not None:
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        signal = service._evaluate_signal(
            strategy=strategy,
            strategy_profile=strategy_profile,
            enriched=enriched,
            indices=current_indices,
            timestamp=ts,
        )
        if signal.action in {Action.LONG, Action.SHORT} and signal.recommended_timing == RecommendedTiming.NOW:
            signals_now += 1
            pending_entry = _PendingEntry(signal=signal)

    if position is not None and trigger_end_idx > 0:
        final_candle = trigger_frame.iloc[trigger_end_idx - 1]
        trades.append(service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"])))

    summary = service._summarize_trades(
        trades=trades,
        strategy_profile=strategy_profile,
        symbol=symbol,
        signals_now=signals_now,
        skipped_entries=skipped_entries,
    )
    return summary, trades


def fmt_metric(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_optional_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row[key]) for key, _ in columns) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def summarize_rows(spec: EntryComboSpec, summary) -> dict[str, Any]:
    return {
        "profile": spec.profile_name,
        "label": spec.label,
        "enabled": spec.enabled_count,
        "reversal": int(spec.include_reversal),
        "regained_fast": int(spec.include_regained_fast),
        "held_slow": int(spec.include_held_slow),
        "auxiliary": int(spec.include_auxiliary),
        "trades": int(summary.total_trades),
        "pf": float(summary.profit_factor),
        "exp_r": float(summary.expectancy_r),
        "cum_r": float(summary.cumulative_r),
        "dd_r": float(summary.max_drawdown_r),
        "signals_now": int(summary.signals_now),
        "skipped_entries": int(summary.skipped_entries),
    }


def compute_average_marginal_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_by_profile = {row["profile"]: row for row in rows}
    specs = [
        EntryComboSpec(
            include_reversal=bool(int(row["reversal"])),
            include_regained_fast=bool(int(row["regained_fast"])),
            include_held_slow=bool(int(row["held_slow"])),
            include_auxiliary=bool(int(row["auxiliary"])),
        )
        for row in rows
    ]
    effects = []

    for field, _ in COMPONENT_FIELDS:
        deltas_cum = []
        deltas_exp = []
        deltas_pf = []
        for spec in specs:
            if spec.component_map[field]:
                continue
            base = row_by_profile[spec.profile_name]
            toggled_map = spec.component_map | {field: True}
            toggled_spec = EntryComboSpec(
                include_reversal=toggled_map["reversal"],
                include_regained_fast=toggled_map["regained_fast"],
                include_held_slow=toggled_map["held_slow"],
                include_auxiliary=toggled_map["auxiliary"],
            )
            if toggled_spec.profile_name not in row_by_profile:
                continue
            alt = row_by_profile[toggled_spec.profile_name]
            deltas_cum.append(float(alt["cum_r"]) - float(base["cum_r"]))
            deltas_exp.append(float(alt["exp_r"]) - float(base["exp_r"]))
            deltas_pf.append(float(alt["pf"]) - float(base["pf"]))
        if not deltas_cum:
            effects.append(
                {
                    "component": field,
                    "avg_delta_cum_r": None,
                    "avg_delta_exp_r": None,
                    "avg_delta_pf": None,
                }
            )
            continue
        effects.append(
            {
                "component": field,
                "avg_delta_cum_r": round(sum(deltas_cum) / len(deltas_cum), 4),
                "avg_delta_exp_r": round(sum(deltas_exp) / len(deltas_exp), 4),
                "avg_delta_pf": round(sum(deltas_pf) / len(deltas_pf), 4),
            }
        )
    return effects


def sort_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            float(item["cum_r"]),
            float(item["pf"]),
            float(item["exp_r"]),
            -float(item["dd_r"]),
        ),
        reverse=True,
    )


def save_outputs(
    *,
    output_dir: Path,
    report: BacktestReport,
    full_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
    oos_trades: list[dict[str, Any]],
    full_effects: list[dict[str, Any]],
    oos_effects: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    BacktestService.save_report(report, output_dir)
    pd.DataFrame(fold_rows).to_csv(output_dir / "entry_attribution_folds.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(output_dir / "entry_attribution_oos_summary.csv", index=False)
    pd.DataFrame(oos_trades).to_csv(output_dir / "entry_attribution_oos_trades.csv", index=False)
    pd.DataFrame(full_effects).to_csv(output_dir / "entry_attribution_full_effects.csv", index=False)
    pd.DataFrame(oos_effects).to_csv(output_dir / "entry_attribution_oos_effects.csv", index=False)

    baseline = next(row for row in oos_rows if row["profile"] == "entry_attr_r1_rf1_hs1_aux1")
    report_md = "\n".join(
        [
            "# Entry Attribution Matrix",
            "",
            f"生成时间：{datetime.now(timezone.utc).isoformat()}",
            "",
            "## 设定",
            "",
            f"- 标的：`{args.symbol}`",
            f"- 总窗口：`{args.start} -> {args.end}`",
            f"- Walk-forward：`{args.train_days}d train / {args.test_days}d test / {args.step_days}d step / {args.scheme}`",
            "- 固定 admission：保留 `gate`",
            "- 拆解对象：`reversal / regained_fast / held_slow / auxiliary`",
            "- Exit：`LONG scaled 1R -> 3R / SHORT fixed 1.5R`",
            "- Swing 模式：`confirmed`",
            "",
            "## OOS Top 8",
            "",
            render_table(
                sort_result_rows(oos_rows)[:8],
                [
                    ("label", "组合"),
                    ("reversal", "R"),
                    ("regained_fast", "RF"),
                    ("held_slow", "HS"),
                    ("auxiliary", "AUX"),
                    ("trades", "OOS交易数"),
                    ("pf", "OOS PF"),
                    ("exp_r", "OOS Exp"),
                    ("cum_r", "OOS累计R"),
                    ("dd_r", "OOS回撤R"),
                ],
            ),
            "",
            "## 当前主线组合",
            "",
            render_table(
                [baseline],
                [
                    ("label", "组合"),
                    ("trades", "OOS交易数"),
                    ("pf", "OOS PF"),
                    ("exp_r", "OOS Exp"),
                    ("cum_r", "OOS累计R"),
                    ("dd_r", "OOS回撤R"),
                ],
            ),
            "",
            "## OOS 平均边际贡献",
            "",
            render_table(
                oos_effects,
                [
                    ("component", "组件"),
                    ("avg_delta_cum_r", "平均Δ累计R"),
                    ("avg_delta_exp_r", "平均ΔExp"),
                    ("avg_delta_pf", "平均ΔPF"),
                ],
            ),
            "",
            "## 全窗口平均边际贡献",
            "",
            render_table(
                full_effects,
                [
                    ("component", "组件"),
                    ("avg_delta_cum_r", "平均Δ累计R"),
                    ("avg_delta_exp_r", "平均ΔExp"),
                    ("avg_delta_pf", "平均ΔPF"),
                ],
            ),
            "",
            "## 原始文件",
            "",
            "- 标准 backtest JSON/CSV：同目录导出",
            "- Fold CSV：`entry_attribution_folds.csv`",
            "- OOS 汇总 CSV：`entry_attribution_oos_summary.csv`",
            "- OOS trades CSV：`entry_attribution_oos_trades.csv`",
            "- 全窗口边际贡献 CSV：`entry_attribution_full_effects.csv`",
            "- OOS 边际贡献 CSV：`entry_attribution_oos_effects.csv`",
        ]
    )
    (output_dir / "entry_attribution_report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "entry_attribution_results.json").write_text(
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
                "full_window": full_rows,
                "walk_forward_folds": fold_rows,
                "walk_forward_oos": oos_rows,
                "full_effects": full_effects,
                "oos_effects": oos_effects,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    start = parse_date(args.start)
    end = parse_date(args.end)
    specs = filter_specs(generate_specs(), args.profiles)
    service = build_service()

    base_frames = service.prepare_history(
        exchange=args.exchange,
        market_type=args.market_type,
        symbol=args.symbol,
        strategy_profile="swing_trend_long_regime_gate_v1",
        start=start,
        end=end,
    )
    base_strategy = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    enriched = {
        timeframe: service._enrich_frame(base_strategy, timeframe, frame)
        for timeframe, frame in base_frames.items()
    }

    full_summaries = []
    full_trades = []
    for spec in specs:
        strategy = build_strategy(spec)
        summary, trades = run_custom_strategy_with_enriched_frames(
            service=service,
            strategy=strategy,
            symbol=args.symbol,
            strategy_profile=spec.profile_name,
            start=start,
            end=end,
            enriched=enriched,
        )
        full_summaries.append(summary)
        full_trades.extend(trades)

    full_report = BacktestReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        exchange=args.exchange,
        market_type=args.market_type,
        start=start.isoformat(),
        end=end.isoformat(),
        symbols=[args.symbol],
        strategy_profiles=[spec.profile_name for spec in specs],
        assumptions=EXIT_ASSUMPTIONS,
        overall=full_summaries,
        by_symbol=full_summaries,
        trades=full_trades,
    )
    full_rows = [summarize_rows(spec, summary) for spec, summary in zip(specs, full_summaries)]

    folds = generate_folds(
        start=start,
        end=end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        scheme=args.scheme,
    )
    fold_rows = []
    oos_rows = []
    oos_trades = []

    for spec in specs:
        strategy = build_strategy(spec)
        profile = spec.profile_name
        all_test_trades = []
        test_signals_now = 0
        test_skipped_entries = 0
        for fold in folds:
            train_summary, _ = run_custom_strategy_with_enriched_frames(
                service=service,
                strategy=strategy,
                symbol=args.symbol,
                strategy_profile=profile,
                start=fold.train_start,
                end=fold.train_end,
                enriched=enriched,
            )
            test_summary, test_trades = run_custom_strategy_with_enriched_frames(
                service=service,
                strategy=strategy,
                symbol=args.symbol,
                strategy_profile=profile,
                start=fold.test_start,
                end=fold.test_end,
                enriched=enriched,
            )
            all_test_trades.extend(test_trades)
            test_signals_now += test_summary.signals_now
            test_skipped_entries += test_summary.skipped_entries
            fold_rows.append(
                {
                    "profile": profile,
                    "label": spec.label,
                    "reversal": int(spec.include_reversal),
                    "regained_fast": int(spec.include_regained_fast),
                    "held_slow": int(spec.include_held_slow),
                    "auxiliary": int(spec.include_auxiliary),
                    "fold": fold.index,
                    "train_start": fold.train_start.date().isoformat(),
                    "train_end": fold.train_end.date().isoformat(),
                    "test_start": fold.test_start.date().isoformat(),
                    "test_end": fold.test_end.date().isoformat(),
                    "train_trades": int(train_summary.total_trades),
                    "train_pf": round(float(train_summary.profit_factor), 4),
                    "train_exp_r": round(float(train_summary.expectancy_r), 4),
                    "train_cum_r": round(float(train_summary.cumulative_r), 4),
                    "test_trades": int(test_summary.total_trades),
                    "test_pf": round(float(test_summary.profit_factor), 4),
                    "test_exp_r": round(float(test_summary.expectancy_r), 4),
                    "test_cum_r": round(float(test_summary.cumulative_r), 4),
                    "test_dd_r": round(float(test_summary.max_drawdown_r), 4),
                }
            )

        oos_summary = service._summarize_trades(
            trades=all_test_trades,
            strategy_profile=profile,
            symbol=args.symbol,
            signals_now=test_signals_now,
            skipped_entries=test_skipped_entries,
        )
        oos_rows.append(summarize_rows(spec, oos_summary))
        oos_trades.extend(asdict(item) for item in all_test_trades)

    full_effects = compute_average_marginal_effects(full_rows)
    oos_effects = compute_average_marginal_effects(oos_rows)
    save_outputs(
        output_dir=Path(args.output_dir),
        report=full_report,
        full_rows=full_rows,
        fold_rows=fold_rows,
        oos_rows=oos_rows,
        oos_trades=oos_trades,
        full_effects=full_effects,
        oos_effects=oos_effects,
        args=args,
    )

    top_oos = sort_result_rows(oos_rows)[:8]
    print("OOS top combos:")
    for row in top_oos:
        print(
            f"  {row['label']}: trades={row['trades']} pf={fmt_metric(row['pf'])} "
            f"exp={fmt_metric(row['exp_r'])} cum_r={fmt_metric(row['cum_r'])} dd_r={fmt_metric(row['dd_r'])}"
        )
    print("OOS average marginal effects:")
    for row in oos_effects:
        print(
            f"  {row['component']}: delta_cum_r={fmt_optional_metric(row['avg_delta_cum_r'])} "
            f"delta_exp_r={fmt_optional_metric(row['avg_delta_exp_r'])} "
            f"delta_pf={fmt_optional_metric(row['avg_delta_pf'])}"
        )
    print(f"Saved outputs under: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
