from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import mean
from typing import Any, Optional

import pandas as pd

from app.backtesting.service import BacktestService, _PendingEntry
from app.schemas.common import Action, Bias, RecommendedTiming, TriggerState, VolatilityState
from app.strategies.scoring import ScoreCard
from app.strategies.windowed_mtf import WindowedMTFStrategy
from app.utils.timeframes import get_strategy_required_timeframes


@dataclass
class SignalDiagnostic:
    timestamp: str
    higher_bias: str
    trend_strength: int
    volatility_state: str
    trend_friendly: bool
    setup_aligned: bool
    setup_execution_ready: bool
    setup_pullback_ready: bool
    setup_reversal_ready: bool
    setup_divergence_ready: bool
    setup_divergence_required: bool
    setup_free_space_ready: bool
    setup_free_space_required: bool
    setup_free_space_r: Optional[float]
    setup_free_space_min_r: float
    setup_structure_ready: bool
    setup_extended: bool
    setup_distance_to_execution_atr: float
    setup_score: int
    trigger_state: str
    trigger_score: int
    trigger_no_new_extreme: bool
    trigger_regained_fast: bool
    trigger_held_slow: bool
    trigger_auxiliary_count: int
    trigger_bullish_rejection: bool
    trigger_bearish_rejection: bool
    trigger_volume_contracting: bool
    confidence: int
    action: str
    bias: str
    recommended_timing: str
    wait_reason: str
    signal_now: bool
    entry_opened: bool = False
    entry_skipped: bool = False


def collect_signal_diagnostics(
    *,
    service: BacktestService,
    exchange: str,
    market_type: str,
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    strategy = service.strategy_service.build_strategy(strategy_profile)
    if not isinstance(strategy, WindowedMTFStrategy):
        raise TypeError(f"Diagnostics currently support WindowedMTFStrategy only, got {strategy_profile}")

    frames = service._load_history(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        strategy_profile=strategy_profile,
        start=start,
        end=end,
    )
    enriched = {
        timeframe: service._enrich_frame(strategy, timeframe, frame)
        for timeframe, frame in frames.items()
    }

    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    required = tuple(get_strategy_required_timeframes(strategy_profile))
    trigger_frame = enriched[trigger_tf]
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}

    min_required = max(int(service.assumptions.lookback // 3), 20)
    position = None
    pending_entry: Optional[_PendingEntry] = None
    cooldown_remaining = 0
    cooldown_bars_after_exit = int(strategy.config.get("backtest", {}).get("cooldown_bars_after_exit", 0))
    signal_lookup: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for trigger_idx in range(len(trigger_frame)):
        candle = trigger_frame.iloc[trigger_idx]
        ts = candle["timestamp"].to_pydatetime()

        if pending_entry is not None:
            maybe_position = service._open_pending_entry(
                symbol=symbol,
                strategy_profile=strategy_profile,
                pending_entry=pending_entry,
                candle=candle,
            )
            signal_key = pending_entry.signal.timestamp.isoformat()
            row = signal_lookup.get(signal_key)
            if row is not None:
                if maybe_position is None:
                    row["entry_skipped"] = True
                else:
                    row["entry_opened"] = True
            position = maybe_position
            pending_entry = None

        if position is not None:
            trade = service._update_open_position(
                position=position,
                candle=candle,
                max_hold_bars=service._max_hold_bars(strategy_profile),
            )
            if trade is not None:
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

        if not ready or any(index < min_required for index in current_indices.values()):
            continue
        if position is not None or pending_entry is not None:
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        row = _evaluate_signal_row(
            service=service,
            strategy=strategy,
            enriched=enriched,
            indices=current_indices,
            timestamp=ts,
        )
        row_dict = asdict(row)
        rows.append(row_dict)
        signal_lookup[row.timestamp] = row_dict

        if row.signal_now:
            signal = service._evaluate_signal(
                strategy=strategy,
                strategy_profile=strategy_profile,
                enriched=enriched,
                indices=current_indices,
                timestamp=ts,
            )
            pending_entry = _PendingEntry(signal=signal)

    return pd.DataFrame(rows)


def build_phase_funnel(signals: pd.DataFrame, *, action_threshold: int) -> list[dict[str, Any]]:
    if signals.empty:
        return []

    setup_execution_ready = (
        signals["setup_execution_ready"]
        if "setup_execution_ready" in signals.columns
        else signals["setup_pullback_ready"]
    )
    setup_free_space_required = (
        signals["setup_free_space_required"]
        if "setup_free_space_required" in signals.columns
        else pd.Series(False, index=signals.index)
    )
    setup_free_space_ready = (
        signals["setup_free_space_ready"]
        if "setup_free_space_ready" in signals.columns
        else pd.Series(True, index=signals.index)
    )
    confirmed_states = {
        TriggerState.BULLISH_CONFIRMED.value,
        TriggerState.BEARISH_CONFIRMED.value,
    }
    stage_masks: list[tuple[str, pd.Series]] = [
        ("evaluated", pd.Series(True, index=signals.index)),
        ("higher_bias_non_neutral", signals["higher_bias"] != Bias.NEUTRAL.value),
        (
            "trend_friendly",
            (signals["higher_bias"] != Bias.NEUTRAL.value) & signals["trend_friendly"],
        ),
        (
            "setup_aligned",
            (signals["higher_bias"] != Bias.NEUTRAL.value) & signals["trend_friendly"] & signals["setup_aligned"],
        ),
        (
            "setup_pullback_ready",
            (signals["higher_bias"] != Bias.NEUTRAL.value)
            & signals["trend_friendly"]
            & signals["setup_aligned"]
            & setup_execution_ready,
        ),
        (
            "setup_reversal_ready",
            (signals["higher_bias"] != Bias.NEUTRAL.value)
            & signals["trend_friendly"]
            & signals["setup_aligned"]
            & setup_execution_ready
            & signals["setup_reversal_ready"],
        ),
        (
            "setup_divergence_ready",
            (signals["higher_bias"] != Bias.NEUTRAL.value)
            & signals["trend_friendly"]
            & signals["setup_aligned"]
            & setup_execution_ready
            & signals["setup_reversal_ready"]
            & ((~signals["setup_divergence_required"]) | signals["setup_divergence_ready"]),
        ),
        (
            "setup_free_space_ready",
            (signals["higher_bias"] != Bias.NEUTRAL.value)
            & signals["trend_friendly"]
            & signals["setup_aligned"]
            & setup_execution_ready
            & signals["setup_reversal_ready"]
            & ((~signals["setup_divergence_required"]) | signals["setup_divergence_ready"])
            & ((~setup_free_space_required) | setup_free_space_ready),
        ),
        (
            "trigger_confirmed",
            (signals["higher_bias"] != Bias.NEUTRAL.value)
            & signals["trend_friendly"]
            & signals["setup_aligned"]
            & setup_execution_ready
            & signals["setup_reversal_ready"]
            & ((~signals["setup_divergence_required"]) | signals["setup_divergence_ready"])
            & ((~setup_free_space_required) | setup_free_space_ready)
            & signals["trigger_state"].isin(confirmed_states),
        ),
        (
            "confidence_pass",
            (signals["higher_bias"] != Bias.NEUTRAL.value)
            & signals["trend_friendly"]
            & signals["setup_aligned"]
            & setup_execution_ready
            & signals["setup_reversal_ready"]
            & ((~signals["setup_divergence_required"]) | signals["setup_divergence_ready"])
            & ((~setup_free_space_required) | setup_free_space_ready)
            & signals["trigger_state"].isin(confirmed_states)
            & (signals["confidence"] >= action_threshold),
        ),
        ("signal_now", signals["signal_now"]),
    ]

    results: list[dict[str, Any]] = []
    previous_count = None
    for name, mask in stage_masks:
        count = int(mask.sum())
        pct_of_all = round((count / len(signals)) * 100, 2)
        pct_of_prev = round((count / previous_count) * 100, 2) if previous_count else 100.0
        results.append(
            {
                "stage": name,
                "count": count,
                "pct_of_all": pct_of_all,
                "pct_of_prev": pct_of_prev,
            }
        )
        previous_count = count
    return results


def summarize_performance(rows: pd.DataFrame, *, group_by: str) -> list[dict[str, Any]]:
    if rows.empty:
        return []

    summaries: list[dict[str, Any]] = []
    for key, group in rows.groupby(group_by, dropna=False):
        pnl = group["pnl_r"].astype(float)
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]
        profit_factor = float(winners.sum() / abs(losers.sum())) if not losers.empty else 0.0
        summaries.append(
            {
                group_by: "NA" if pd.isna(key) else str(key),
                "count": int(len(group)),
                "win_rate": round(float((pnl > 0).mean() * 100), 2),
                "avg_r": round(float(pnl.mean()), 4),
                "cumulative_r": round(float(pnl.sum()), 4),
                "profit_factor": round(profit_factor, 4),
            }
        )
    return summaries


def bucket_confidence(confidence: int) -> str:
    if confidence < 65:
        return "<65"
    if confidence < 75:
        return "65-74"
    if confidence < 85:
        return "75-84"
    if confidence < 95:
        return "85-94"
    return "95+"


def bucket_distance(distance_atr: float) -> str:
    if distance_atr <= 0:
        return "inside_execution_zone"
    if distance_atr <= 0.25:
        return "0-0.25ATR"
    if distance_atr <= 0.5:
        return "0.25-0.5ATR"
    if distance_atr <= 1.0:
        return "0.5-1.0ATR"
    return ">1.0ATR"


def derive_findings(
    *,
    summary: dict[str, Any],
    trades: pd.DataFrame,
    phase_funnel: list[dict[str, Any]],
) -> list[str]:
    findings: list[str] = []
    if trades.empty:
        return ["没有成交单，当前样本不足以诊断。"]

    side_stats = trades.groupby("side")["pnl_r"].agg(["count", "mean", "sum"])
    if {"LONG", "SHORT"}.issubset(side_stats.index):
        long_sum = float(side_stats.loc["LONG", "sum"])
        short_sum = float(side_stats.loc["SHORT", "sum"])
        if long_sum < 0 < short_sum:
            findings.append(
                f"主要拖累来自多头而不是空头：LONG 累计 {long_sum:.2f}R，SHORT 累计 {short_sum:.2f}R。"
            )

    quarter_stats = trades.groupby("quarter")["pnl_r"].sum().sort_values(ascending=False)
    if not quarter_stats.empty:
        top_quarter = quarter_stats.index[0]
        top_quarter_sum = float(quarter_stats.iloc[0])
        total_sum = float(trades["pnl_r"].sum())
        if total_sum != 0 and top_quarter_sum > abs(total_sum) * 0.8:
            findings.append(
                f"收益高度集中在单一阶段：{top_quarter} 贡献 {top_quarter_sum:.2f}R，说明策略稳定性不足。"
            )

    exit_counts = trades["exit_reason"].value_counts()
    stop_count = int(exit_counts.get("stop_loss", 0))
    target_count = int(sum(count for reason, count in exit_counts.items() if str(reason).startswith("take_profit_")))
    if stop_count > target_count:
        findings.append(
            f"止损次数 {stop_count} 高于止盈次数 {target_count}，说明当前 entry 过滤质量仍然不够高。"
        )

    confidence_buckets = trades["confidence_bucket"].nunique(dropna=True)
    if confidence_buckets <= 1:
        findings.append("成交单几乎全部落在同一置信度桶，当前置信度分层对筛选优劣单没有帮助。")

    funnel_map = {item["stage"]: item for item in phase_funnel}
    if "setup_divergence_ready" in funnel_map:
        setup_gate_stage = "setup_divergence_ready"
    elif "setup_reversal_ready" in funnel_map:
        setup_gate_stage = "setup_reversal_ready"
    else:
        setup_gate_stage = "setup_pullback_ready"
    if setup_gate_stage in funnel_map and "signal_now" in funnel_map:
        ready_count = int(funnel_map[setup_gate_stage]["count"])
        now_count = int(funnel_map["signal_now"]["count"])
        if ready_count and now_count / ready_count < 0.5:
            if setup_gate_stage == "setup_divergence_ready":
                findings.append("大量机会在 1H 位置、reversal K 线和 divergence 都合格后，仍被触发确认拦掉，trigger 层有明显收缩。")
            elif setup_gate_stage == "setup_reversal_ready":
                findings.append("大量机会在 1H 位置和 reversal K 线都合格后，仍被触发确认拦掉，trigger 层有明显收缩。")
            else:
                findings.append("大量机会在 1H 位置合格后，仍被 1H 触发确认拦掉，触发层有明显收缩。")

    expectancy = float(summary["expectancy_r"])
    if expectancy > 0 and expectancy < 0.1:
        findings.append("总体已转正，但期望值只有很薄的一层安全垫，交易成本和参数漂移都足以把它打回负值。")
    elif expectancy <= 0:
        findings.append("总体仍未转正，当前漏斗没有形成足够厚的正向边际。")

    return findings


def _evaluate_signal_row(
    *,
    service: BacktestService,
    strategy: WindowedMTFStrategy,
    enriched: dict[str, pd.DataFrame],
    indices: dict[str, int],
    timestamp: datetime,
) -> SignalDiagnostic:
    prepared = {
        timeframe: service._build_snapshot(strategy, timeframe, enriched[timeframe], indices[timeframe])
        for timeframe in indices
    }

    higher_bias, trend_strength = strategy._derive_higher_timeframe_bias(prepared)
    setup_key = str(strategy.window_config["setup_timeframe"])
    trigger_key = str(strategy.window_config["trigger_timeframe"])
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

    reference_key = str(strategy.window_config.get("reference_timeframe", setup_key))
    setup_assessment = strategy._assess_setup(
        higher_bias,
        prepared[setup_key],
        setup_key,
        reference_ctx=prepared[reference_key],
        current_price=prepared[trigger_key].model.close,
    )
    trigger_assessment = strategy._assess_trigger(
        higher_bias,
        prepared[trigger_key],
        trigger_key,
        trend_strength=trend_strength,
    )
    scorecard.add(setup_assessment["score"], f"{setup_key}_setup", setup_assessment["score_note"])
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
    signal_now = action in {Action.LONG, Action.SHORT} and recommended_timing == RecommendedTiming.NOW

    return SignalDiagnostic(
        timestamp=timestamp.isoformat(),
        higher_bias=higher_bias.value,
        trend_strength=int(trend_strength),
        volatility_state=volatility_state.value,
        trend_friendly=is_trend_friendly,
        setup_aligned=bool(setup_assessment["aligned"]),
        setup_execution_ready=bool(setup_assessment.get("execution_ready", setup_assessment["pullback_ready"])),
        setup_pullback_ready=bool(setup_assessment["pullback_ready"]),
        setup_divergence_ready=bool(setup_assessment.get("divergence_ready", True)),
        setup_divergence_required=bool(setup_assessment.get("require_divergence_gate", False)),
        setup_free_space_ready=bool(setup_assessment.get("free_space_ready", True)),
        setup_free_space_required=bool(setup_assessment.get("require_free_space_gate", False)),
        setup_free_space_r=(
            float(setup_assessment["free_space_r"]) if setup_assessment.get("free_space_r") is not None else None
        ),
        setup_free_space_min_r=float(setup_assessment.get("free_space_min_r", 0.0)),
        setup_structure_ready=bool(setup_assessment["structure_ready"]),
        setup_extended=bool(setup_assessment["is_extended"]),
        setup_distance_to_execution_atr=float(setup_assessment["distance_to_value_atr"]),
        setup_score=int(setup_assessment["score"]),
        trigger_state=trigger_assessment["state"].value,
        trigger_score=int(trigger_assessment["score"]),
        trigger_no_new_extreme=bool(trigger_assessment.get("no_new_extreme", False)),
        trigger_regained_fast=bool(trigger_assessment.get("regained_fast", False)),
        trigger_held_slow=bool(trigger_assessment.get("held_slow", False)),
        trigger_auxiliary_count=int(trigger_assessment.get("auxiliary_count", 0)),
        trigger_bullish_rejection=bool(trigger_assessment.get("bullish_rejection", False)),
        trigger_bearish_rejection=bool(trigger_assessment.get("bearish_rejection", False)),
        trigger_volume_contracting=bool(trigger_assessment.get("volume_contracting", False)),
        confidence=int(confidence),
        action=action.value,
        bias=bias.value,
        recommended_timing=recommended_timing.value,
        wait_reason=_classify_wait_reason(
            higher_bias=higher_bias,
            trend_friendly=is_trend_friendly,
            setup_assessment=setup_assessment,
            trigger_assessment=trigger_assessment,
            confidence=confidence,
            action=action,
            recommended_timing=recommended_timing,
            action_threshold=int(strategy.config["confidence"]["action_threshold"]),
        ),
        signal_now=signal_now,
        setup_reversal_ready=bool(setup_assessment.get("reversal_ready", True)),
    )


def _classify_wait_reason(
    *,
    higher_bias: Bias,
    trend_friendly: bool,
    setup_assessment: dict[str, Any],
    trigger_assessment: dict[str, Any],
    confidence: int,
    action: Action,
    recommended_timing: RecommendedTiming,
    action_threshold: int,
) -> str:
    if action in {Action.LONG, Action.SHORT} and recommended_timing == RecommendedTiming.NOW:
        return "enter_now"
    if higher_bias == Bias.NEUTRAL:
        return "higher_timeframe_neutral"
    if not trend_friendly:
        return "trend_not_friendly"
    if not setup_assessment["aligned"]:
        return "setup_misaligned"
    if setup_assessment["is_extended"]:
        return "setup_extended"
    if (
        setup_assessment.get("execution_ready", setup_assessment["pullback_ready"])
        and setup_assessment.get("require_free_space_gate")
        and not setup_assessment.get("free_space_ready", True)
    ):
        return "setup_free_space_not_ready"
    if not setup_assessment["pullback_ready"]:
        if not setup_assessment["structure_ready"]:
            return "setup_structure_not_ready"
        return "setup_not_in_execution_zone"
    if setup_assessment.get("require_reversal_candle") and not setup_assessment.get("reversal_ready", True):
        return "setup_reversal_not_ready"
    if setup_assessment.get("require_divergence_gate") and not setup_assessment.get("divergence_ready", True):
        return "setup_divergence_not_ready"
    if trigger_assessment["state"] == TriggerState.MIXED:
        return "trigger_mixed"
    if trigger_assessment["state"] == TriggerState.NONE:
        return "trigger_none"
    if confidence < action_threshold:
        return "confidence_below_threshold"
    return "wait_other"
