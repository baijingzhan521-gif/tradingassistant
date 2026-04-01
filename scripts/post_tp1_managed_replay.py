from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestTrade, _PendingEntry
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Action, RecommendedTiming
from app.services.strategy_service import StrategyService


EXIT_ASSUMPTIONS = {
    "exit_profile": "post_tp1_extension_matrix",
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

FLOAT_TOLERANCE = 1e-4

BASELINE_PROFILE_LABELS = {
    "swing_trend_long_regime_gate_v1": "Champion Mainline",
    "swing_trend_long_regime_short_no_reversal_no_aux_v1": "Challenger Short No Reversal + No Auxiliary",
}


@dataclass(frozen=True)
class ExtensionProfileSpec:
    name: str
    label: str
    mode: str
    observation_bars: int = 0
    veto_hold_trend_min: int | None = None
    veto_hold_trend_max: int | None = None


PROFILE_SPECS = (
    ExtensionProfileSpec("baseline_be_after_tp1", "Baseline: BE After TP1", "baseline"),
    ExtensionProfileSpec(
        "hold_structure_after_tp1_all_longs",
        "Hold Structure After TP1: All LONGs",
        "hold_structure",
    ),
    ExtensionProfileSpec(
        "be_if_no_extension_within_2bars_after_tp1",
        "BE If No 1H Extension Within 2 Bars After TP1",
        "extension",
        observation_bars=2,
    ),
    ExtensionProfileSpec(
        "be_if_no_extension_within_3bars_after_tp1",
        "BE If No 1H Extension Within 3 Bars After TP1",
        "extension",
        observation_bars=3,
    ),
    ExtensionProfileSpec(
        "be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98",
        "3 Bars, But Revert To BE If Entry Trend Is 96-98",
        "extension",
        observation_bars=3,
        veto_hold_trend_min=96,
        veto_hold_trend_max=98,
    ),
)

PROFILE_SPEC_MAP = {spec.name: spec for spec in PROFILE_SPECS}


@dataclass
class LongExtensionState:
    observation_bars: int
    tp1_hit_idx: int | None = None
    tp1_hit_bar_high: float | None = None
    observed_bars: int = 0
    decision_made: bool = False
    extension_confirmed: bool | None = None
    decision_trigger_idx: int | None = None
    decision_reason: str | None = None
    last_checked_idx: int | None = None
    last_checked_high: float | None = None


def baseline_profile_label(strategy_profile: str) -> str:
    return BASELINE_PROFILE_LABELS.get(strategy_profile, strategy_profile)


def build_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = dict(EXIT_ASSUMPTIONS)
    if assumption_overrides:
        assumptions.update(assumption_overrides)
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**assumptions),
    )


def trades_to_frame(trades: list[BacktestTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
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
        )
    frame = pd.DataFrame([asdict(trade) for trade in trades])
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values("entry_time").reset_index(drop=True)


def precompute_extension_features(*, trigger_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": trigger_frame["high"].astype(float),
            "close": trigger_frame["close"].astype(float),
        }
    )


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
        same_exit = abs(float(row["exit_price_baseline"]) - float(row["exit_price_simulated"])) <= FLOAT_TOLERANCE
        same_r = abs(float(row["pnl_r_baseline"]) - float(row["pnl_r_simulated"])) <= FLOAT_TOLERANCE
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


def open_position_for_profile(
    *,
    service: BacktestService,
    pending_entry: _PendingEntry,
    candle: pd.Series,
    symbol: str,
    strategy_profile: str,
    spec: ExtensionProfileSpec,
):
    maybe_position = service._open_pending_entry(
        symbol=symbol,
        strategy_profile=strategy_profile,
        pending_entry=pending_entry,
        candle=candle,
    )
    if maybe_position is None:
        return None, None
    state = None
    if maybe_position.side == Action.LONG:
        if spec.mode == "baseline":
            maybe_position.move_stop_to_entry_after_tp1 = True
        else:
            maybe_position.move_stop_to_entry_after_tp1 = False
        if spec.mode == "extension":
            state = LongExtensionState(observation_bars=spec.observation_bars)
    return maybe_position, state


def update_extension_state(
    *,
    position,
    state: LongExtensionState,
    spec: ExtensionProfileSpec,
    current_trigger_idx: int,
    extension_features: pd.DataFrame,
) -> None:
    if state.decision_made or state.tp1_hit_idx is None or state.tp1_hit_bar_high is None:
        return
    last_closed_idx = current_trigger_idx - 1
    if last_closed_idx <= state.tp1_hit_idx:
        return
    feature_row = extension_features.iloc[last_closed_idx]
    current_high = float(feature_row["high"])
    state.last_checked_idx = last_closed_idx
    state.last_checked_high = current_high
    state.observed_bars += 1

    if current_high > float(state.tp1_hit_bar_high):
        veto_hold = (
            spec.veto_hold_trend_min is not None
            and spec.veto_hold_trend_max is not None
            and spec.veto_hold_trend_min <= int(position.trend_strength) <= spec.veto_hold_trend_max
        )
        if veto_hold:
            position.current_stop_price = max(position.current_stop_price, position.entry_price)
            state.decision_made = True
            state.extension_confirmed = False
            state.decision_trigger_idx = current_trigger_idx
            state.decision_reason = (
                f"extension_veto_trend_{spec.veto_hold_trend_min}_{spec.veto_hold_trend_max}"
            )
            return

        state.decision_made = True
        state.extension_confirmed = True
        state.decision_trigger_idx = current_trigger_idx
        state.decision_reason = "extension_confirmed"
        return

    if state.observed_bars >= state.observation_bars:
        position.current_stop_price = max(position.current_stop_price, position.entry_price)
        state.decision_made = True
        state.extension_confirmed = False
        state.decision_trigger_idx = current_trigger_idx
        state.decision_reason = "no_extension_within_window"


def run_profile(
    *,
    service: BacktestService,
    strategy,
    symbol: str,
    strategy_profile: str,
    spec: ExtensionProfileSpec,
    start: datetime,
    end: datetime,
    enriched: dict[str, pd.DataFrame],
    extension_features: pd.DataFrame,
) -> tuple[Any, list[BacktestTrade], list[dict[str, Any]]]:
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    required = tuple(strategy.required_timeframes)
    trigger_frame = enriched[trigger_tf]
    trigger_end_idx = int(trigger_frame["timestamp"].searchsorted(pd.Timestamp(end), side="left"))
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}

    trades: list[BacktestTrade] = []
    annotated_rows: list[dict[str, Any]] = []
    pending_entry: _PendingEntry | None = None
    position = None
    long_extension_state: LongExtensionState | None = None
    signals_now = 0
    skipped_entries = 0
    cooldown_remaining = 0
    cooldown_bars_after_exit = int(strategy.config.get("backtest", {}).get("cooldown_bars_after_exit", 0))

    for trigger_idx in range(trigger_end_idx):
        candle = trigger_frame.iloc[trigger_idx]
        ts = candle["timestamp"].to_pydatetime()

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

        if pending_entry is not None:
            maybe_position, maybe_state = open_position_for_profile(
                service=service,
                pending_entry=pending_entry,
                candle=candle,
                symbol=symbol,
                strategy_profile=strategy_profile,
                spec=spec,
            )
            if maybe_position is None:
                skipped_entries += 1
            else:
                position = maybe_position
                long_extension_state = maybe_state
            pending_entry = None

        if position is not None:
            if spec.mode == "extension" and position.side == Action.LONG and long_extension_state is not None:
                update_extension_state(
                    position=position,
                    state=long_extension_state,
                    spec=spec,
                    current_trigger_idx=trigger_idx,
                    extension_features=extension_features,
                )

            trade = service._update_open_position(
                position=position,
                candle=candle,
                max_hold_bars=service._max_hold_bars(strategy_profile),
            )
            if trade is not None:
                annotation = {
                    "profile": spec.name,
                    "profile_label": spec.label,
                    "extension_observation_bars": spec.observation_bars if spec.mode == "extension" else None,
                    "extension_tp1_hit_idx": long_extension_state.tp1_hit_idx if long_extension_state is not None else None,
                    "extension_tp1_hit_bar_high": long_extension_state.tp1_hit_bar_high if long_extension_state is not None else None,
                    "extension_observed_bars": long_extension_state.observed_bars if long_extension_state is not None else None,
                    "extension_decision_made": long_extension_state.decision_made if long_extension_state is not None else None,
                    "extension_confirmed": long_extension_state.extension_confirmed if long_extension_state is not None else None,
                    "extension_decision_reason": long_extension_state.decision_reason if long_extension_state is not None else None,
                    "extension_last_checked_idx": long_extension_state.last_checked_idx if long_extension_state is not None else None,
                    "extension_last_checked_high": long_extension_state.last_checked_high if long_extension_state is not None else None,
                }
                trades.append(trade)
                annotated_rows.append({**asdict(trade), **annotation})
                position = None
                long_extension_state = None
                cooldown_remaining = cooldown_bars_after_exit
            elif spec.mode == "extension" and position.side == Action.LONG and long_extension_state is not None:
                if position.tp1_hit and long_extension_state.tp1_hit_idx is None:
                    long_extension_state.tp1_hit_idx = trigger_idx
                    long_extension_state.tp1_hit_bar_high = float(candle["high"])

        if ts < start:
            continue
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
        trade = service._close_position(
            position,
            final_candle,
            exit_reason="end_of_test",
            fill_price=float(final_candle["close"]),
        )
        annotation = {
            "profile": spec.name,
            "profile_label": spec.label,
            "extension_observation_bars": spec.observation_bars if spec.mode == "extension" else None,
            "extension_tp1_hit_idx": long_extension_state.tp1_hit_idx if long_extension_state is not None else None,
            "extension_tp1_hit_bar_high": long_extension_state.tp1_hit_bar_high if long_extension_state is not None else None,
            "extension_observed_bars": long_extension_state.observed_bars if long_extension_state is not None else None,
            "extension_decision_made": long_extension_state.decision_made if long_extension_state is not None else None,
            "extension_confirmed": long_extension_state.extension_confirmed if long_extension_state is not None else None,
            "extension_decision_reason": long_extension_state.decision_reason if long_extension_state is not None else None,
            "extension_last_checked_idx": long_extension_state.last_checked_idx if long_extension_state is not None else None,
            "extension_last_checked_high": long_extension_state.last_checked_high if long_extension_state is not None else None,
        }
        trades.append(trade)
        annotated_rows.append({**asdict(trade), **annotation})

    summary = service._summarize_trades(
        trades=trades,
        strategy_profile=strategy_profile,
        symbol=symbol,
        signals_now=signals_now,
        skipped_entries=skipped_entries,
    )
    return summary, trades, annotated_rows


def summarize_side_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    results: list[dict[str, Any]] = []
    group_cols = [
        "window",
        "baseline_strategy_profile",
        "baseline_profile_label",
        "baseline_source",
        "profile",
        "label",
        "side",
    ]
    for keys, group in rows.groupby(group_cols, sort=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = {column: value for column, value in zip(group_cols, keys)}
        gross_profit = float(group.loc[group["pnl_r"] > 0, "pnl_r"].sum())
        gross_loss = abs(float(group.loc[group["pnl_r"] < 0, "pnl_r"].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        row.update(
            {
                "trades": int(len(group)),
                "cum_r": round(float(group["pnl_r"].sum()), 4),
                "avg_r": round(float(group["pnl_r"].mean()), 4),
                "profit_factor": round(float(profit_factor), 4),
                "tp2_hit_rate_pct": round(float(group["tp2_hit"].mean() * 100), 2),
                "stop_loss_rate_pct": round(float((group["exit_reason"] == "stop_loss").mean() * 100), 2),
            }
        )
        results.append(row)
    return results
