from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestReport, _PendingEntry
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Action, Bias, RecommendedTiming
from app.services.strategy_service import StrategyService
from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    SwingTrendLongRegimeGateV1Strategy,
)


EXIT_ASSUMPTIONS = {
    "exit_profile": "stop_ablation_long_scaled1_3_short_fixed1_5",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}

STOP_MODES = (
    ("structure", "Structure + ATR"),
    ("ema21", "EMA21 Initial Stop"),
    ("ema55", "EMA55 Initial Stop"),
    ("hybrid_ema55_cap", "Hybrid: Cap At EMA55"),
    ("structure_ema21_trail_after_tp1", "Structure + EMA21 Trail After TP1"),
)

WINDOW_PRESETS = {
    "two_year": ("2024-03-19", "2026-03-19"),
    "full_2020": ("2020-03-19", "2026-03-19"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stop ablation on the BTC mainline.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument(
        "--windows",
        default="two_year,full_2020",
        help="Comma-separated presets: two_year,full_2020",
    )
    parser.add_argument(
        "--stop-modes",
        default=",".join(mode for mode, _ in STOP_MODES),
        help="Comma-separated stop modes",
    )
    parser.add_argument("--output-dir", default="artifacts/backtests/stop_ablation_mainline")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_service() -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**EXIT_ASSUMPTIONS),
    )


class StopAblationStrategy(SwingTrendLongRegimeGateV1Strategy):
    required_timeframes = ("1d", "4h", "1h")

    def __init__(self, *, stop_mode: str, profile_name: str) -> None:
        super().__init__(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
        self.stop_mode = stop_mode
        self.name = profile_name

    def _derive_trade_levels(
        self,
        *,
        bias: Bias,
        setup_ctx,
        reference_ctx,
        current_price: float,
    ) -> dict[str, Any]:
        base = super()._derive_trade_levels(
            bias=bias,
            setup_ctx=setup_ctx,
            reference_ctx=reference_ctx,
            current_price=current_price,
        )
        if self.stop_mode in {"structure", "structure_ema21_trail_after_tp1"}:
            return base

        epsilon = max(float(setup_ctx.model.atr14) * 0.05, 1e-6)
        ema21 = float(setup_ctx.model.ema21)
        ema55 = float(setup_ctx.model.ema55)
        stop_price = float(base["stop_price"])

        if bias == Bias.BULLISH:
            if self.stop_mode == "ema21":
                stop_price = ema21
            elif self.stop_mode == "ema55":
                stop_price = ema55
            elif self.stop_mode == "hybrid_ema55_cap":
                stop_price = max(float(base["stop_price"]), ema55)
            if stop_price >= current_price - epsilon:
                stop_price = min(float(base["stop_price"]), current_price - epsilon)
            risk = max(current_price - stop_price, epsilon)
            tp1 = current_price + risk
            tp2 = current_price + (2 * risk)
            reference_target = float(reference_ctx.model.swing_high) if reference_ctx.model.swing_high is not None else None
            if reference_target is not None:
                tp2 = max(tp2, reference_target)
            free_space_r = None if reference_target is None or risk <= 0 else (reference_target - current_price) / risk
            return {
                **base,
                "stop_price": stop_price,
                "risk": risk,
                "tp1": tp1,
                "tp2": tp2,
                "reference_target": reference_target,
                "free_space_r": free_space_r,
                "invalidation_price": float(setup_ctx.model.swing_low or base["entry_low"]),
            }

        if self.stop_mode == "ema21":
            stop_price = ema21
        elif self.stop_mode == "ema55":
            stop_price = ema55
        elif self.stop_mode == "hybrid_ema55_cap":
            stop_price = min(float(base["stop_price"]), ema55)
        if stop_price <= current_price + epsilon:
            stop_price = max(float(base["stop_price"]), current_price + epsilon)
        risk = max(stop_price - current_price, epsilon)
        tp1 = current_price - risk
        tp2 = current_price - (2 * risk)
        reference_target = float(reference_ctx.model.swing_low) if reference_ctx.model.swing_low is not None else None
        if reference_target is not None:
            tp2 = min(tp2, reference_target)
        free_space_r = None if reference_target is None or risk <= 0 else (current_price - reference_target) / risk
        return {
            **base,
            "stop_price": stop_price,
            "risk": risk,
            "tp1": tp1,
            "tp2": tp2,
            "reference_target": reference_target,
            "free_space_r": free_space_r,
            "invalidation_price": float(setup_ctx.model.swing_high or base["entry_high"]),
        }


def run_custom_strategy_with_enriched_frames(
    *,
    service: BacktestService,
    strategy: StopAblationStrategy,
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
    enriched: dict[str, pd.DataFrame],
) -> tuple[Any, list[Any]]:
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
            # Use previous closed-bar EMA21 to avoid same-bar lookahead.
            if strategy.stop_mode == "structure_ema21_trail_after_tp1" and position.tp1_hit and trigger_idx > 0:
                prev_candle = trigger_frame.iloc[trigger_idx - 1]
                ema21_prev = float(prev_candle["ema21"])
                if position.side == Action.LONG:
                    position.current_stop_price = max(position.current_stop_price, ema21_prev)
                else:
                    position.current_stop_price = min(position.current_stop_price, ema21_prev)

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
        trades.append(
            service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"]))
        )

    summary = service._summarize_trades(
        trades=trades,
        strategy_profile=strategy_profile,
        symbol=symbol,
        signals_now=signals_now,
        skipped_entries=skipped_entries,
    )
    return summary, trades


def compute_extra_metrics(trades: list[Any]) -> dict[str, Any]:
    if not trades:
        return {
            "avg_stop_distance_pct": 0.0,
            "median_stop_distance_pct": 0.0,
            "stop_loss_rate_pct": 0.0,
            "breakeven_after_tp1_rate_pct": 0.0,
        }
    rows = pd.DataFrame(asdict(trade) for trade in trades)
    stop_distance_pct = (rows["entry_price"] - rows["stop_price"]).abs() / rows["entry_price"] * 100
    return {
        "avg_stop_distance_pct": round(float(stop_distance_pct.mean()), 4),
        "median_stop_distance_pct": round(float(stop_distance_pct.median()), 4),
        "stop_loss_rate_pct": round(float((rows["exit_reason"] == "stop_loss").mean() * 100), 2),
        "breakeven_after_tp1_rate_pct": round(float((rows["exit_reason"] == "breakeven_after_tp1").mean() * 100), 2),
    }


def summarize_side_rows(trades: list[Any]) -> list[dict[str, Any]]:
    if not trades:
        return []
    rows = pd.DataFrame(asdict(trade) for trade in trades)
    result = []
    for side, group in rows.groupby("side"):
        wins = int((group["pnl_r"] > 0).sum())
        losses = int((group["pnl_r"] < 0).sum())
        gross_profit = float(group.loc[group["pnl_r"] > 0, "pnl_r"].sum())
        gross_loss = abs(float(group.loc[group["pnl_r"] < 0, "pnl_r"].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        result.append(
            {
                "side": side,
                "trades": int(len(group)),
                "wins": wins,
                "losses": losses,
                "cum_r": round(float(group["pnl_r"].sum()), 4),
                "avg_r": round(float(group["pnl_r"].mean()), 4),
                "profit_factor": round(float(profit_factor), 4),
                "stop_loss_rate_pct": round(float((group["exit_reason"] == "stop_loss").mean() * 100), 2),
            }
        )
    return result


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
    body = []
    for row in rows:
        parts = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                parts.append(f"{value:.4f}")
            else:
                parts.append(str(value))
        body.append("| " + " | ".join(parts) + " |")
    return "\n".join([header, divider, *body])


def main() -> None:
    args = parse_args()
    configure_logging()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_windows = [item.strip() for item in args.windows.split(",") if item.strip()]
    unknown_windows = sorted(set(selected_windows) - set(WINDOW_PRESETS))
    if unknown_windows:
        raise ValueError(f"Unsupported windows: {', '.join(unknown_windows)}")

    stop_mode_labels = dict(STOP_MODES)
    selected_modes = [item.strip() for item in args.stop_modes.split(",") if item.strip()]
    unknown_modes = sorted(set(selected_modes) - set(stop_mode_labels))
    if unknown_modes:
        raise ValueError(f"Unsupported stop modes: {', '.join(unknown_modes)}")

    service = build_service()
    all_summary_rows: list[dict[str, Any]] = []
    all_side_rows: list[dict[str, Any]] = []
    report_sections = [
        "# Stop Ablation Report",
        "",
        "- 基线 entry 家族固定为 `swing_trend_long_regime_gate_v1`。",
        "- 本次不是逐笔静态重算，而是每种 stop 模式都重放完整 backtest 序列，所以 exit 改变会真实影响后续可交易机会。",
        "- `structure+EMA21 trailing` 使用的是 `TP1` 之后、按前一根已完成 1H K 线 `EMA21` 更新 stop，避免 same-bar lookahead。",
        "",
    ]

    for window_name in selected_windows:
        start_raw, end_raw = WINDOW_PRESETS[window_name]
        start = parse_date(start_raw)
        end = parse_date(end_raw)
        base_frames = service.prepare_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile="swing_trend_long_regime_gate_v1",
            start=start,
            end=end,
        )
        base_strategy = service.strategy_service.build_strategy("swing_trend_long_regime_gate_v1")
        enriched = {
            timeframe: service._enrich_frame(base_strategy, timeframe, frame)
            for timeframe, frame in base_frames.items()
        }

        window_rows = []
        for stop_mode in selected_modes:
            profile_name = f"stop_ablation_{stop_mode}"
            strategy = StopAblationStrategy(stop_mode=stop_mode, profile_name=profile_name)
            summary, trades = run_custom_strategy_with_enriched_frames(
                service=service,
                strategy=strategy,
                symbol=args.symbol,
                strategy_profile=profile_name,
                start=start,
                end=end,
                enriched=enriched,
            )
            extra = compute_extra_metrics(trades)
            row = {
                "window": window_name,
                "start": start_raw,
                "end": end_raw,
                "stop_mode": stop_mode,
                "label": stop_mode_labels[stop_mode],
                "trades": int(summary.total_trades),
                "win_rate_pct": round(float(summary.win_rate), 2),
                "profit_factor": round(float(summary.profit_factor), 4),
                "expectancy_r": round(float(summary.expectancy_r), 4),
                "cum_r": round(float(summary.cumulative_r), 4),
                "max_dd_r": round(float(summary.max_drawdown_r), 4),
                "avg_holding_bars": round(float(summary.avg_holding_bars), 2),
                "tp1_hit_rate_pct": round(float(summary.tp1_hit_rate), 2),
                "tp2_hit_rate_pct": round(float(summary.tp2_hit_rate), 2),
                "signals_now": int(summary.signals_now),
                "skipped_entries": int(summary.skipped_entries),
                **extra,
            }
            window_rows.append(row)
            all_summary_rows.append(row)

            side_rows = summarize_side_rows(trades)
            for side_row in side_rows:
                all_side_rows.append(
                    {
                        "window": window_name,
                        "stop_mode": stop_mode,
                        "label": stop_mode_labels[stop_mode],
                        **side_row,
                    }
                )

            trade_rows = [asdict(trade) for trade in trades]
            if trade_rows:
                write_csv(output_dir / f"{window_name}_{stop_mode}_trades.csv", trade_rows)

        window_rows.sort(key=lambda item: (item["cum_r"], item["profit_factor"]), reverse=True)
        write_csv(output_dir / f"{window_name}_summary.csv", window_rows)
        window_side_rows = [row for row in all_side_rows if row["window"] == window_name]
        write_csv(output_dir / f"{window_name}_side_summary.csv", window_side_rows)

        report_sections.extend(
            [
                f"## {window_name}",
                "",
                markdown_table(
                    window_rows,
                    [
                        ("label", "Stop Mode"),
                        ("trades", "Trades"),
                        ("win_rate_pct", "Win Rate %"),
                        ("profit_factor", "PF"),
                        ("expectancy_r", "Exp R"),
                        ("cum_r", "Cum R"),
                        ("max_dd_r", "Max DD R"),
                        ("avg_stop_distance_pct", "Avg Stop Dist %"),
                        ("stop_loss_rate_pct", "Stop Loss %"),
                        ("breakeven_after_tp1_rate_pct", "BE After TP1 %"),
                    ],
                ),
                "",
                "按方向拆开：",
                "",
                markdown_table(
                    window_side_rows,
                    [
                        ("label", "Stop Mode"),
                        ("side", "Side"),
                        ("trades", "Trades"),
                        ("cum_r", "Cum R"),
                        ("avg_r", "Avg R"),
                        ("profit_factor", "PF"),
                        ("stop_loss_rate_pct", "Stop Loss %"),
                    ],
                ),
                "",
            ]
        )

    write_csv(output_dir / "summary_all.csv", all_summary_rows)
    write_csv(output_dir / "side_summary_all.csv", all_side_rows)
    (output_dir / "report.md").write_text("\n".join(report_sections).strip() + "\n", encoding="utf-8")

    print(f"Saved report: {output_dir / 'report.md'}")
    print(f"Saved summary CSV: {output_dir / 'summary_all.csv'}")
    print(f"Saved side summary CSV: {output_dir / 'side_summary_all.csv'}")


if __name__ == "__main__":
    main()
