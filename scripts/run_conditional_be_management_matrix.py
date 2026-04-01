from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import (
    BacktestAssumptions,
    BacktestService,
    BacktestSummary,
    BacktestTrade,
    _PendingEntry,
    _SignalSnapshot,
)
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Action, Bias, RecommendedTiming, VolatilityState
from app.services.strategy_service import StrategyService
from app.strategies.scoring import ScoreCard


EXIT_ASSUMPTIONS = {
    "exit_profile": "conditional_be_management_matrix",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {
        "take_profit_mode": "scaled",
        "scaled_tp1_r": 1.0,
        "scaled_tp2_r": 3.0,
        "move_stop_to_entry_after_tp1": True,
    },
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}

WINDOW_PRESETS = {
    "two_year": ("2024-03-19", "2026-03-19"),
    "full_2020": ("2020-03-19", "2026-03-19"),
}


@dataclass(frozen=True)
class ConditionalProfileSpec:
    name: str
    label: str
    min_trend_strength: int | None = None
    min_free_space_r: float | None = None
    disable_be_for_all_longs: bool = False

    def disable_be(self, *, side: Action, trend_strength: int, free_space_r: float | None) -> bool:
        if side != Action.LONG:
            return False
        if self.disable_be_for_all_longs:
            return True
        if self.min_trend_strength is None and self.min_free_space_r is None:
            return False
        if self.min_trend_strength is not None and trend_strength < self.min_trend_strength:
            return False
        if self.min_free_space_r is not None and (free_space_r is None or free_space_r < self.min_free_space_r):
            return False
        return True


PROFILE_SPECS = (
    ConditionalProfileSpec("baseline_be_after_tp1", "Baseline: BE After TP1"),
    ConditionalProfileSpec(
        "hold_structure_after_tp1_all_longs",
        "Hold Structure After TP1: All LONGs",
        disable_be_for_all_longs=True,
    ),
    ConditionalProfileSpec(
        "no_be_if_long_trend_ge_95",
        "No BE If LONG Trend >= 95",
        min_trend_strength=95,
    ),
    ConditionalProfileSpec(
        "no_be_if_long_trend_ge_98",
        "No BE If LONG Trend >= 98",
        min_trend_strength=98,
    ),
    ConditionalProfileSpec(
        "no_be_if_long_free_space_ge_3",
        "No BE If LONG Free Space >= 3R",
        min_free_space_r=3.0,
    ),
    ConditionalProfileSpec(
        "no_be_if_long_free_space_ge_4",
        "No BE If LONG Free Space >= 4R",
        min_free_space_r=4.0,
    ),
    ConditionalProfileSpec(
        "no_be_if_long_trend_ge_95_and_free_space_ge_3",
        "No BE If LONG Trend >= 95 And Free Space >= 3R",
        min_trend_strength=95,
        min_free_space_r=3.0,
    ),
    ConditionalProfileSpec(
        "no_be_if_long_trend_ge_95_and_free_space_ge_4",
        "No BE If LONG Trend >= 95 And Free Space >= 4R",
        min_trend_strength=95,
        min_free_space_r=4.0,
    ),
    ConditionalProfileSpec(
        "no_be_if_long_trend_ge_98_and_free_space_ge_3",
        "No BE If LONG Trend >= 98 And Free Space >= 3R",
        min_trend_strength=98,
        min_free_space_r=3.0,
    ),
    ConditionalProfileSpec(
        "no_be_if_long_trend_ge_98_and_free_space_ge_4",
        "No BE If LONG Trend >= 98 And Free Space >= 4R",
        min_trend_strength=98,
        min_free_space_r=4.0,
    ),
)


@dataclass
class SignalEnvelope:
    signal: _SignalSnapshot
    trend_strength: int
    confidence: int
    higher_bias: str
    free_space_r: float | None
    free_space_ready: bool
    volatility_state: str


@dataclass
class PendingSignalEnvelope:
    signal: _SignalSnapshot
    trend_strength: int
    confidence: int
    higher_bias: str
    free_space_r: float | None
    free_space_ready: bool
    volatility_state: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conditional post-TP1 BE management matrix.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument(
        "--windows",
        default="two_year,full_2020",
        help="Comma-separated presets: two_year,full_2020",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(spec.name for spec in PROFILE_SPECS),
        help="Comma-separated profile names",
    )
    parser.add_argument(
        "--baseline-dir",
        default="artifacts/backtests/stop_ablation_mainline",
        help="Directory containing *_structure_trades.csv baseline artifacts.",
    )
    parser.add_argument("--output-dir", default="artifacts/backtests/conditional_be_management_mainline")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_service() -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**EXIT_ASSUMPTIONS),
    )


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
    body: list[str] = []
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def load_baseline_trades(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path)
    if trades.empty:
        raise ValueError(f"Baseline trades CSV is empty: {path}")
    for column in ("signal_time", "entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    return trades.sort_values("entry_time").reset_index(drop=True)


def evaluate_signal_with_context(
    *,
    service: BacktestService,
    strategy,
    enriched: dict[str, pd.DataFrame],
    indices: dict[str, int],
    timestamp: datetime,
) -> SignalEnvelope:
    prepared = {
        timeframe: service._build_snapshot(strategy, timeframe, enriched[timeframe], indices[timeframe])
        for timeframe in indices
    }

    higher_bias, trend_strength = strategy._derive_higher_timeframe_bias(prepared)
    setup_key = str(strategy.window_config["setup_timeframe"])
    trigger_key = str(strategy.window_config["trigger_timeframe"])
    reference_key = str(strategy.window_config.get("reference_timeframe", setup_key))

    volatility_state = strategy._derive_volatility_state(prepared[setup_key])
    is_trend_friendly = strategy._is_trend_friendly(
        higher_bias=higher_bias,
        trend_strength=trend_strength,
        volatility_state=volatility_state,
    )

    scorecard = ScoreCard(base=50)
    higher_label = strategy._format_timeframe_group(tuple(strategy.window_config["higher_timeframes"]))
    if higher_bias == Bias.BULLISH:
        scorecard.add(15, "higher_bias", f"{higher_label} 同步偏多")
    elif higher_bias == Bias.BEARISH:
        scorecard.add(15, "higher_bias", f"{higher_label} 同步偏空")
    else:
        scorecard.add(-20, "higher_conflict", f"{higher_label} 没有形成同向共振")

    setup_assessment = strategy._assess_setup(
        higher_bias,
        prepared[setup_key],
        setup_key,
        reference_ctx=prepared[reference_key],
        current_price=prepared[trigger_key].model.close,
    )
    scorecard.add(setup_assessment["score"], f"{setup_key}_setup", setup_assessment["score_note"])

    trigger_assessment = strategy._assess_trigger(
        higher_bias,
        prepared[trigger_key],
        trigger_key,
        trend_strength=trend_strength,
    )
    scorecard.add(trigger_assessment["score"], f"{trigger_key}_trigger", trigger_assessment["score_note"])

    if volatility_state == VolatilityState.HIGH:
        scorecard.add(-15, "volatility", f"{setup_key} ATR 百分比偏高")
    elif volatility_state == VolatilityState.LOW:
        scorecard.add(3, "volatility", f"{setup_key} 波动可控")

    confidence = scorecard.total
    action, bias, recommended_timing = strategy._decide(
        higher_bias=higher_bias,
        trend_friendly=is_trend_friendly,
        setup_assessment=setup_assessment,
        trigger_assessment=trigger_assessment,
        confidence=confidence,
    )
    trade_plan = strategy._build_trade_plan(
        action=action,
        bias=bias,
        setup_ctx=prepared[setup_key],
        reference_ctx=prepared[reference_key],
        current_price=prepared[trigger_key].model.close,
        setup_key=setup_key,
        reference_key=reference_key,
    )
    signal = _SignalSnapshot(
        action=action,
        bias=bias,
        trend_strength=trend_strength,
        confidence=confidence,
        recommended_timing=recommended_timing,
        entry_zone_low=float(trade_plan["entry_zone"].low) if trade_plan["entry_zone"] else None,
        entry_zone_high=float(trade_plan["entry_zone"].high) if trade_plan["entry_zone"] else None,
        stop_price=float(trade_plan["stop_loss"].price) if trade_plan["stop_loss"] else None,
        tp1_price=float(trade_plan["take_profit_hint"].tp1) if trade_plan["take_profit_hint"] else None,
        tp2_price=float(trade_plan["take_profit_hint"].tp2) if trade_plan["take_profit_hint"] else None,
        invalidation_price=float(trade_plan["invalidation_price"]) if trade_plan["invalidation_price"] is not None else None,
        timestamp=timestamp,
    )
    free_space_value = setup_assessment.get("free_space_r")
    return SignalEnvelope(
        signal=signal,
        trend_strength=int(trend_strength),
        confidence=int(confidence),
        higher_bias=higher_bias.value,
        free_space_r=float(free_space_value) if free_space_value is not None else None,
        free_space_ready=bool(setup_assessment.get("free_space_ready", True)),
        volatility_state=volatility_state.value,
    )


def run_profile(
    *,
    service: BacktestService,
    strategy,
    symbol: str,
    strategy_profile: str,
    profile_spec: ConditionalProfileSpec,
    start: datetime,
    end: datetime,
    enriched: dict[str, pd.DataFrame],
) -> tuple[BacktestSummary, list[BacktestTrade], list[dict[str, Any]]]:
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    required = tuple(strategy.required_timeframes)
    trigger_frame = enriched[trigger_tf]
    trigger_end_idx = int(trigger_frame["timestamp"].searchsorted(pd.Timestamp(end), side="left"))
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}

    trades: list[BacktestTrade] = []
    annotated_rows: list[dict[str, Any]] = []
    pending_entry: PendingSignalEnvelope | None = None
    current_annotation: dict[str, Any] | None = None
    position = None
    signals_now = 0
    skipped_entries = 0
    cooldown_remaining = 0
    cooldown_bars_after_exit = int(strategy.config.get("backtest", {}).get("cooldown_bars_after_exit", 0))
    no_be_long_signals = 0
    long_signals_now = 0
    no_be_long_trades = 0

    for trigger_idx in range(trigger_end_idx):
        candle = trigger_frame.iloc[trigger_idx]
        ts = candle["timestamp"].to_pydatetime()

        if pending_entry is not None:
            maybe_position = service._open_pending_entry(
                symbol=symbol,
                strategy_profile=strategy_profile,
                pending_entry=_PendingEntry(signal=pending_entry.signal),
                candle=candle,
            )
            if maybe_position is None:
                skipped_entries += 1
            else:
                disable_be = profile_spec.disable_be(
                    side=maybe_position.side,
                    trend_strength=pending_entry.trend_strength,
                    free_space_r=pending_entry.free_space_r,
                )
                if maybe_position.side == Action.LONG:
                    maybe_position.move_stop_to_entry_after_tp1 = not disable_be
                    if disable_be:
                        no_be_long_trades += 1
                position = maybe_position
                current_annotation = {
                    "profile": profile_spec.name,
                    "profile_label": profile_spec.label,
                    "signal_free_space_r": pending_entry.free_space_r,
                    "signal_free_space_ready": pending_entry.free_space_ready,
                    "signal_trend_strength": pending_entry.trend_strength,
                    "signal_confidence": pending_entry.confidence,
                    "signal_higher_bias": pending_entry.higher_bias,
                    "signal_volatility_state": pending_entry.volatility_state,
                    "long_no_be_applied": disable_be,
                }
            pending_entry = None

        if position is not None:
            trade = service._update_open_position(
                position=position,
                candle=candle,
                max_hold_bars=service._max_hold_bars(strategy_profile),
            )
            if trade is not None:
                trades.append(trade)
                annotated_rows.append({**asdict(trade), **(current_annotation or {})})
                position = None
                current_annotation = None
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

        envelope = evaluate_signal_with_context(
            service=service,
            strategy=strategy,
            enriched=enriched,
            indices=current_indices,
            timestamp=ts,
        )
        signal = envelope.signal
        if signal.action in {Action.LONG, Action.SHORT} and signal.recommended_timing == RecommendedTiming.NOW:
            signals_now += 1
            if signal.action == Action.LONG:
                long_signals_now += 1
                if profile_spec.disable_be(
                    side=signal.action,
                    trend_strength=envelope.trend_strength,
                    free_space_r=envelope.free_space_r,
                ):
                    no_be_long_signals += 1
            pending_entry = PendingSignalEnvelope(
                signal=signal,
                trend_strength=envelope.trend_strength,
                confidence=envelope.confidence,
                higher_bias=envelope.higher_bias,
                free_space_r=envelope.free_space_r,
                free_space_ready=envelope.free_space_ready,
                volatility_state=envelope.volatility_state,
            )

    if position is not None and trigger_end_idx > 0:
        final_candle = trigger_frame.iloc[trigger_end_idx - 1]
        trade = service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"]))
        trades.append(trade)
        annotated_rows.append({**asdict(trade), **(current_annotation or {})})

    summary = service._summarize_trades(
        trades=trades,
        strategy_profile=strategy_profile,
        symbol=symbol,
        signals_now=signals_now,
        skipped_entries=skipped_entries,
    )
    summary_row = {
        **asdict(summary),
        "long_signals_now": long_signals_now,
        "no_be_long_signals": no_be_long_signals,
        "no_be_long_trades": no_be_long_trades,
    }
    return BacktestSummary(**{k: summary_row[k] for k in asdict(summary)}), trades, annotated_rows


def validate_baseline_replay(baseline: pd.DataFrame, simulated: pd.DataFrame) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    subset = simulated[
        [
            "signal_time",
            "entry_time",
            "exit_time",
            "side",
            "exit_reason",
            "exit_price",
            "tp1_hit",
            "tp2_hit",
            "pnl_r",
        ]
    ].copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        subset[column] = pd.to_datetime(subset[column], utc=True)
    merged = baseline.merge(
        subset,
        on=["signal_time", "entry_time", "side"],
        suffixes=("_baseline", "_simulated"),
        how="inner",
    )
    if len(merged) != len(baseline):
        raise ValueError("Baseline replay validation failed: simulated trade count does not match baseline count.")
    for _, row in merged.iterrows():
        same_reason = str(row["exit_reason_baseline"]) == str(row["exit_reason_simulated"])
        same_tp1 = bool(row["tp1_hit_baseline"]) == bool(row["tp1_hit_simulated"])
        same_tp2 = bool(row["tp2_hit_baseline"]) == bool(row["tp2_hit_simulated"])
        same_exit = abs(float(row["exit_price_baseline"]) - float(row["exit_price_simulated"])) <= 1e-4
        same_r = abs(float(row["pnl_r_baseline"]) - float(row["pnl_r_simulated"])) <= 1e-4
        if not (same_reason and same_tp1 and same_tp2 and same_exit and same_r):
            mismatches.append(
                {
                    "signal_time": pd.Timestamp(row["signal_time"]).isoformat(),
                    "side": row["side"],
                    "baseline_exit_reason": row["exit_reason_baseline"],
                    "simulated_exit_reason": row["exit_reason_simulated"],
                    "baseline_exit_price": round(float(row["exit_price_baseline"]), 6),
                    "simulated_exit_price": round(float(row["exit_price_simulated"]), 6),
                    "baseline_pnl_r": round(float(row["pnl_r_baseline"]), 6),
                    "simulated_pnl_r": round(float(row["pnl_r_simulated"]), 6),
                }
            )
    return mismatches


def summarize_side_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for (window, profile, label, side), group in rows.groupby(["window", "profile", "label", "side"], sort=False):
        wins = int((group["pnl_r"] > 0).sum())
        losses = int((group["pnl_r"] < 0).sum())
        gross_profit = float(group.loc[group["pnl_r"] > 0, "pnl_r"].sum())
        gross_loss = abs(float(group.loc[group["pnl_r"] < 0, "pnl_r"].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        results.append(
            {
                "window": window,
                "profile": profile,
                "label": label,
                "side": side,
                "trades": int(len(group)),
                "cum_r": round(float(group["pnl_r"].sum()), 4),
                "avg_r": round(float(group["pnl_r"].mean()), 4),
                "profit_factor": round(float(profit_factor), 4),
                "tp2_hit_rate_pct": round(float(group["tp2_hit"].mean() * 100), 2),
                "stop_loss_rate_pct": round(float((group["exit_reason"] == "stop_loss").mean() * 100), 2),
                "no_be_applied_trades": int(group.get("long_no_be_applied", pd.Series(dtype=bool)).fillna(False).sum())
                if side == "LONG"
                else 0,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    configure_logging()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = ROOT / args.baseline_dir

    selected_windows = [item.strip() for item in args.windows.split(",") if item.strip()]
    selected_profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    profile_map = {spec.name: spec for spec in PROFILE_SPECS}

    unknown_windows = sorted(set(selected_windows) - set(WINDOW_PRESETS))
    if unknown_windows:
        raise ValueError(f"Unsupported windows: {', '.join(unknown_windows)}")
    unknown_profiles = sorted(set(selected_profiles) - set(profile_map))
    if unknown_profiles:
        raise ValueError(f"Unsupported profiles: {', '.join(unknown_profiles)}")

    service = build_service()
    strategy = service.strategy_service.build_strategy("swing_trend_long_regime_gate_v1")

    all_summary_rows: list[dict[str, Any]] = []
    all_side_rows: list[dict[str, Any]] = []
    all_trade_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    report_sections = [
        "# Conditional BE Management Matrix",
        "",
        "- 这次直接跑 sequence-aware 完整 backtest，不再只做固定 entry 诊断。",
        "- 变化只作用在 `LONG` 的 `TP1` 之后：满足条件时，不把 stop 抬到进场价；否则维持当前 `BE after TP1`。",
        "- 候选条件只用当前代码里已有、且可解释的信号：`trend_strength` 与 `free_space_r`。",
        "- `free_space_r` 来自 setup assessment，表示从当前价到参考高点大约还剩多少 `R` 空间。",
        "",
    ]

    for window_name in selected_windows:
        start_raw, end_raw = WINDOW_PRESETS[window_name]
        start = parse_date(start_raw)
        end = parse_date(end_raw)
        baseline_path = baseline_dir / f"{window_name}_structure_trades.csv"
        baseline_trades = load_baseline_trades(baseline_path)

        base_frames = service.prepare_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile="swing_trend_long_regime_gate_v1",
            start=start,
            end=end,
        )
        enriched = {
            timeframe: service._enrich_frame(strategy, timeframe, frame)
            for timeframe, frame in base_frames.items()
        }

        window_rows: list[dict[str, Any]] = []
        window_trade_rows: list[dict[str, Any]] = []
        baseline_simulated: pd.DataFrame | None = None

        for profile_name in selected_profiles:
            spec = profile_map[profile_name]
            strategy_profile = f"conditional_be_{spec.name}"
            summary, trades, annotated_rows = run_profile(
                service=service,
                strategy=strategy,
                symbol=args.symbol,
                strategy_profile=strategy_profile,
                profile_spec=spec,
                start=start,
                end=end,
                enriched=enriched,
            )
            trade_rows = []
            for row in annotated_rows:
                trade_rows.append({"window": window_name, "profile": spec.name, "label": spec.label, **row})
            write_csv(output_dir / f"{window_name}_{spec.name}_trades.csv", trade_rows)
            window_trade_rows.extend(trade_rows)
            all_trade_rows.extend(trade_rows)

            long_trades = [trade for trade in trades if trade.side == "LONG"]
            no_be_long_trades = sum(1 for row in trade_rows if bool(row.get("long_no_be_applied")))
            row = {
                "window": window_name,
                "profile": spec.name,
                "label": spec.label,
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
                "long_trades": int(len(long_trades)),
                "no_be_long_trades": int(no_be_long_trades),
                "no_be_long_pct": round((no_be_long_trades / len(long_trades) * 100), 2) if long_trades else 0.0,
                "avg_no_be_trend_strength": round(
                    float(pd.Series([row["signal_trend_strength"] for row in trade_rows if row.get("long_no_be_applied")]).mean()),
                    2,
                )
                if no_be_long_trades
                else None,
                "avg_no_be_free_space_r": round(
                    float(
                        pd.Series(
                            [
                                row["signal_free_space_r"]
                                for row in trade_rows
                                if row.get("long_no_be_applied") and row.get("signal_free_space_r") is not None
                            ]
                        ).mean()
                    ),
                    4,
                )
                if any(row.get("long_no_be_applied") and row.get("signal_free_space_r") is not None for row in trade_rows)
                else None,
            }
            window_rows.append(row)
            all_summary_rows.append(row)

            if spec.name == "baseline_be_after_tp1":
                baseline_simulated = pd.DataFrame(trade_rows)

        if baseline_simulated is None:
            raise ValueError("baseline_be_after_tp1 must be included for validation.")
        mismatches = validate_baseline_replay(baseline_trades, baseline_simulated)
        validation_rows.append(
            {
                "window": window_name,
                "baseline_trades": int(len(baseline_trades)),
                "baseline_replay_mismatches": int(len(mismatches)),
            }
        )
        if mismatches:
            mismatch_path = output_dir / f"{window_name}_baseline_validation_mismatches.csv"
            write_csv(mismatch_path, mismatches)
            raise ValueError(f"Baseline replay validation failed for {window_name}: see {mismatch_path}")

        side_rows = summarize_side_rows(pd.DataFrame(window_trade_rows))
        all_side_rows.extend(side_rows)
        write_csv(output_dir / f"{window_name}_summary.csv", window_rows)
        write_csv(output_dir / f"{window_name}_side_summary.csv", side_rows)

        window_rows_sorted = sorted(window_rows, key=lambda item: (item["cum_r"], item["profit_factor"]), reverse=True)
        side_rows_sorted = sorted(side_rows, key=lambda item: (item["label"], item["side"]))

        report_sections.extend(
            [
                f"## {window_name}",
                "",
                markdown_table(
                    window_rows_sorted,
                    [
                        ("label", "Profile"),
                        ("trades", "Trades"),
                        ("profit_factor", "PF"),
                        ("expectancy_r", "Exp R"),
                        ("cum_r", "Cum R"),
                        ("max_dd_r", "Max DD R"),
                        ("no_be_long_trades", "No-BE LONG Trades"),
                        ("no_be_long_pct", "No-BE LONG %"),
                    ],
                ),
                "",
                "按方向拆开：",
                "",
                markdown_table(
                    side_rows_sorted,
                    [
                        ("label", "Profile"),
                        ("side", "Side"),
                        ("trades", "Trades"),
                        ("cum_r", "Cum R"),
                        ("avg_r", "Avg R"),
                        ("profit_factor", "PF"),
                        ("tp2_hit_rate_pct", "TP2 Hit %"),
                        ("no_be_applied_trades", "No-BE Trades"),
                    ],
                ),
                "",
            ]
        )

    write_csv(output_dir / "summary_all.csv", all_summary_rows)
    write_csv(output_dir / "side_summary_all.csv", all_side_rows)
    write_csv(output_dir / "validation.csv", validation_rows)
    write_csv(output_dir / "trades_all.csv", all_trade_rows)
    (output_dir / "report.md").write_text("\n".join(report_sections).strip() + "\n", encoding="utf-8")

    print(f"Saved report: {output_dir / 'report.md'}")
    print(f"Saved summary CSV: {output_dir / 'summary_all.csv'}")
    print(f"Saved side summary CSV: {output_dir / 'side_summary_all.csv'}")
    print(f"Saved validation CSV: {output_dir / 'validation.csv'}")


if __name__ == "__main__":
    main()
