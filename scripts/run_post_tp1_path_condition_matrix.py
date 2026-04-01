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
)
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Action, RecommendedTiming
from app.services.strategy_service import StrategyService


EXIT_ASSUMPTIONS = {
    "exit_profile": "post_tp1_path_condition_matrix",
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
class PathProfileSpec:
    name: str
    label: str
    mode: str
    observation_bars: int = 0
    guard_post_approval_4h_loss: bool = False


PROFILE_SPECS = (
    PathProfileSpec("baseline_be_after_tp1", "Baseline: BE After TP1", "baseline"),
    PathProfileSpec(
        "hold_structure_after_tp1_all_longs",
        "Hold Structure After TP1: All LONGs",
        "hold_structure",
    ),
    PathProfileSpec(
        "path_hold_if_2bars_above_ema21_and_4h_bullish",
        "Path Hold: 2x 1H Close >= EMA21 + 4H Bullish",
        "path",
        observation_bars=2,
    ),
    PathProfileSpec(
        "path_hold_if_3bars_above_ema21_and_4h_bullish",
        "Path Hold: 3x 1H Close >= EMA21 + 4H Bullish",
        "path",
        observation_bars=3,
    ),
    PathProfileSpec(
        "path_hold_if_3bars_above_ema21_and_4h_bullish_then_be_on_4h_loss",
        "Path Hold: 3x 1H Close >= EMA21 + 4H Bullish, Then BE On 4H Loss",
        "path",
        observation_bars=3,
        guard_post_approval_4h_loss=True,
    ),
)


@dataclass
class LongPathState:
    observation_bars: int
    tp1_hit_idx: int | None = None
    observed_bars: int = 0
    decision_made: bool = False
    condition_met: bool | None = None
    decision_trigger_idx: int | None = None
    fail_reason: str | None = None
    last_checked_idx: int | None = None
    last_checked_close: float | None = None
    last_checked_ema21: float | None = None
    last_checked_4h_bullish: bool | None = None
    post_approval_guard_triggered: bool = False
    post_approval_guard_trigger_idx: int | None = None
    post_approval_guard_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-TP1 path-condition management matrix.")
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
    parser.add_argument("--output-dir", default="artifacts/backtests/post_tp1_path_condition_mainline")
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


def infer_bullish(value: Any) -> bool:
    text = str(value).lower()
    return "bullish" in text and "bearish" not in text


def precompute_path_features(*, trigger_frame: pd.DataFrame, four_h_frame: pd.DataFrame) -> pd.DataFrame:
    four_h_timestamps = four_h_frame["timestamp"]
    four_h_indices = []
    for ts in trigger_frame["timestamp"]:
        idx = int(four_h_timestamps.searchsorted(ts, side="right")) - 1
        four_h_indices.append(idx if idx >= 0 else -1)
    four_h_bullish = [
        bool(idx >= 0 and infer_bullish(four_h_frame.iloc[idx]["trend_bias"]))
        for idx in four_h_indices
    ]
    result = pd.DataFrame(
        {
            "close_above_ema21": (trigger_frame["close"] >= trigger_frame["ema21"]).astype(bool),
            "four_h_idx": four_h_indices,
            "four_h_bullish": four_h_bullish,
            "close": trigger_frame["close"].astype(float),
            "ema21": trigger_frame["ema21"].astype(float),
        }
    )
    return result


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


def open_position_for_profile(*, service: BacktestService, pending_entry: _PendingEntry, candle: pd.Series, symbol: str, strategy_profile: str, spec: PathProfileSpec):
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
        if spec.mode == "path":
            state = LongPathState(observation_bars=spec.observation_bars)
    return maybe_position, state


def update_path_state(
    *,
    spec: PathProfileSpec,
    position,
    state: LongPathState,
    current_trigger_idx: int,
    path_features: pd.DataFrame,
) -> None:
    if state.tp1_hit_idx is None:
        return
    last_closed_idx = current_trigger_idx - 1
    if last_closed_idx <= state.tp1_hit_idx:
        return
    feature_row = path_features.iloc[last_closed_idx]
    state.observed_bars += 1
    state.last_checked_idx = last_closed_idx
    state.last_checked_close = float(feature_row["close"])
    state.last_checked_ema21 = float(feature_row["ema21"])
    state.last_checked_4h_bullish = bool(feature_row["four_h_bullish"])

    if not state.decision_made:
        state.observed_bars += 1

        if not bool(feature_row["close_above_ema21"]):
            position.current_stop_price = max(position.current_stop_price, position.entry_price)
            state.decision_made = True
            state.condition_met = False
            state.decision_trigger_idx = current_trigger_idx
            state.fail_reason = "close_below_ema21"
            return

        if not bool(feature_row["four_h_bullish"]):
            position.current_stop_price = max(position.current_stop_price, position.entry_price)
            state.decision_made = True
            state.condition_met = False
            state.decision_trigger_idx = current_trigger_idx
            state.fail_reason = "4h_not_bullish"
            return

        if state.observed_bars >= state.observation_bars:
            state.decision_made = True
            state.condition_met = True
            state.decision_trigger_idx = current_trigger_idx
            state.fail_reason = None
        return

    if not state.condition_met or not spec.guard_post_approval_4h_loss:
        return
    if state.post_approval_guard_triggered:
        return
    if state.decision_trigger_idx is None or last_closed_idx < state.decision_trigger_idx:
        return
    if not bool(feature_row["four_h_bullish"]):
        position.current_stop_price = max(position.current_stop_price, position.entry_price)
        state.post_approval_guard_triggered = True
        state.post_approval_guard_trigger_idx = current_trigger_idx
        state.post_approval_guard_reason = "4h_lost_bullish_after_hold"


def run_profile(
    *,
    service: BacktestService,
    strategy,
    symbol: str,
    strategy_profile: str,
    spec: PathProfileSpec,
    start: datetime,
    end: datetime,
    enriched: dict[str, pd.DataFrame],
    path_features: pd.DataFrame,
) -> tuple[BacktestSummary, list[BacktestTrade], list[dict[str, Any]]]:
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    required = tuple(strategy.required_timeframes)
    trigger_frame = enriched[trigger_tf]
    trigger_end_idx = int(trigger_frame["timestamp"].searchsorted(pd.Timestamp(end), side="left"))
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}

    trades: list[BacktestTrade] = []
    annotated_rows: list[dict[str, Any]] = []
    pending_entry: _PendingEntry | None = None
    position = None
    long_path_state: LongPathState | None = None
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
                long_path_state = maybe_state
            pending_entry = None

        if position is not None:
            if spec.mode == "path" and position.side == Action.LONG and long_path_state is not None:
                update_path_state(
                    spec=spec,
                    position=position,
                    state=long_path_state,
                    current_trigger_idx=trigger_idx,
                    path_features=path_features,
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
                    "path_observation_bars": spec.observation_bars if spec.mode == "path" else None,
                    "path_tp1_hit_idx": long_path_state.tp1_hit_idx if long_path_state is not None else None,
                    "path_observed_bars": long_path_state.observed_bars if long_path_state is not None else None,
                    "path_decision_made": long_path_state.decision_made if long_path_state is not None else None,
                    "path_condition_met": long_path_state.condition_met if long_path_state is not None else None,
                    "path_fail_reason": long_path_state.fail_reason if long_path_state is not None else None,
                    "path_last_checked_idx": long_path_state.last_checked_idx if long_path_state is not None else None,
                    "path_last_checked_close": long_path_state.last_checked_close if long_path_state is not None else None,
                    "path_last_checked_ema21": long_path_state.last_checked_ema21 if long_path_state is not None else None,
                    "path_last_checked_4h_bullish": long_path_state.last_checked_4h_bullish if long_path_state is not None else None,
                    "path_post_approval_guard_triggered": long_path_state.post_approval_guard_triggered if long_path_state is not None else None,
                    "path_post_approval_guard_trigger_idx": long_path_state.post_approval_guard_trigger_idx if long_path_state is not None else None,
                    "path_post_approval_guard_reason": long_path_state.post_approval_guard_reason if long_path_state is not None else None,
                }
                trades.append(trade)
                annotated_rows.append({**asdict(trade), **annotation})
                position = None
                long_path_state = None
                cooldown_remaining = cooldown_bars_after_exit
            elif spec.mode == "path" and position.side == Action.LONG and long_path_state is not None:
                if position.tp1_hit and long_path_state.tp1_hit_idx is None:
                    long_path_state.tp1_hit_idx = trigger_idx

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
        trade = service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"]))
        annotation = {
            "profile": spec.name,
            "profile_label": spec.label,
            "path_observation_bars": spec.observation_bars if spec.mode == "path" else None,
            "path_tp1_hit_idx": long_path_state.tp1_hit_idx if long_path_state is not None else None,
            "path_observed_bars": long_path_state.observed_bars if long_path_state is not None else None,
            "path_decision_made": long_path_state.decision_made if long_path_state is not None else None,
            "path_condition_met": long_path_state.condition_met if long_path_state is not None else None,
            "path_fail_reason": long_path_state.fail_reason if long_path_state is not None else None,
            "path_last_checked_idx": long_path_state.last_checked_idx if long_path_state is not None else None,
            "path_last_checked_close": long_path_state.last_checked_close if long_path_state is not None else None,
            "path_last_checked_ema21": long_path_state.last_checked_ema21 if long_path_state is not None else None,
            "path_last_checked_4h_bullish": long_path_state.last_checked_4h_bullish if long_path_state is not None else None,
            "path_post_approval_guard_triggered": long_path_state.post_approval_guard_triggered if long_path_state is not None else None,
            "path_post_approval_guard_trigger_idx": long_path_state.post_approval_guard_trigger_idx if long_path_state is not None else None,
            "path_post_approval_guard_reason": long_path_state.post_approval_guard_reason if long_path_state is not None else None,
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
        "# Post-TP1 Path Condition Matrix",
        "",
        "- 这次直接跑 sequence-aware 完整 backtest。",
        "- 基线仍是当前 `BE after TP1`。",
        "- 路径条件的操作化定义：`TP1` 之后先观察后续已收 `1H` K；若前 `N` 根都 `close >= EMA21`，且对应时刻最新 `4H trend_bias` 仍是 `bullish`，则继续 hold 原结构止损；只要其中任何一根失败，则从下一根开始切回 `BE`。",
        "- 新增 guard 版：放行之后若后续最新 `4H trend_bias` 不再 `bullish`，也会在下一根把 stop 收到 `BE`。",
        "- 这里的 `4H bullish` 使用 enriched 4H 框架里的 `trend_bias` 字段，不额外发明新判定。",
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
        path_features = precompute_path_features(trigger_frame=enriched["1h"], four_h_frame=enriched["4h"])

        window_rows: list[dict[str, Any]] = []
        window_trade_rows: list[dict[str, Any]] = []
        baseline_simulated: pd.DataFrame | None = None

        for profile_name in selected_profiles:
            spec = profile_map[profile_name]
            strategy_profile = f"path_condition_{spec.name}"
            summary, trades, annotated_rows = run_profile(
                service=service,
                strategy=strategy,
                symbol=args.symbol,
                strategy_profile=strategy_profile,
                spec=spec,
                start=start,
                end=end,
                enriched=enriched,
                path_features=path_features,
            )
            trade_rows = [{"window": window_name, "profile": spec.name, "label": spec.label, **row} for row in annotated_rows]
            write_csv(output_dir / f"{window_name}_{spec.name}_trades.csv", trade_rows)
            window_trade_rows.extend(trade_rows)
            all_trade_rows.extend(trade_rows)

            long_df = pd.DataFrame([row for row in trade_rows if row["side"] == "LONG"])
            path_condition_met_count = (
                int(long_df["path_condition_met"].eq(True).sum()) if not long_df.empty else 0
            )
            path_decision_count = (
                int(long_df["path_decision_made"].eq(True).sum()) if not long_df.empty else 0
            )
            path_guard_trigger_count = (
                int(long_df["path_post_approval_guard_triggered"].eq(True).sum()) if not long_df.empty else 0
            )
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
                "long_trades": int(len(long_df)),
                "path_decision_count": path_decision_count,
                "path_condition_met_count": path_condition_met_count,
                "path_condition_met_pct": round(path_condition_met_count / len(long_df) * 100, 2) if len(long_df) else 0.0,
                "path_guard_trigger_count": path_guard_trigger_count,
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
                        ("path_decision_count", "Path Decisions"),
                        ("path_condition_met_count", "Path Holds"),
                        ("path_condition_met_pct", "Path Hold %"),
                        ("path_guard_trigger_count", "Guard Triggers"),
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
