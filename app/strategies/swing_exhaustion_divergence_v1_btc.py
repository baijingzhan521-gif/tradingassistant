from __future__ import annotations

from typing import Any

from app.schemas.analysis import EntryZone, StopLoss, TakeProfitHint
from app.schemas.common import Action, Bias, RecommendedTiming, TriggerState, VolatilityState
from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, PreparedTimeframe, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_reversal_candle": False,
        "require_structure_ready": False,
    },
    "divergence": {
        **COMMON_DEFAULT_CONFIG["divergence"],
        "enabled": True,
        "min_level": 2,
    },
    "exhaustion": {
        "counter_trend_block_threshold": 90,
        "atr_stop_buffer": 0.18,
        "minimum_r_multiple": 0.8,
        "entry_pad_atr": 0.08,
    },
    "confidence": {"action_threshold": 58},
    "window": {
        "higher_timeframes": ["1d", "4h"],
        "setup_timeframe": "1h",
        "trigger_timeframe": "1h",
        "reference_timeframe": "4h",
        "display_timeframes": ["1d", "4h", "1h"],
        "chart_timeframes": ["1d", "4h", "1h"],
        "trend_strength_threshold": 0,
    },
}


class SwingExhaustionDivergenceV1BTCStrategy(WindowedMTFStrategy):
    name = "swing_exhaustion_divergence_v1_btc"
    required_timeframes = ("1d", "4h", "1h")

    def _derive_higher_timeframe_bias(self, prepared: dict[str, PreparedTimeframe]) -> tuple[Bias, int]:
        trigger_ctx = prepared["1h"]
        day_ctx = prepared["1d"]
        reference_ctx = prepared["4h"]
        divergence = trigger_ctx.divergence_profile
        trend_strength = int(round((day_ctx.model.trend_score + reference_ctx.model.trend_score) / 2))

        bullish_level = int(divergence.get("bullish_level", 0))
        bearish_level = int(divergence.get("bearish_level", 0))
        bullish_signal = bool(divergence.get("bullish_signal", False))
        bearish_signal = bool(divergence.get("bearish_signal", False))
        min_level = int(self.config.get("divergence", {}).get("min_level", 2))

        if bullish_signal and bullish_level >= min_level and not bearish_signal:
            return Bias.BULLISH, trend_strength
        if bearish_signal and bearish_level >= min_level and not bullish_signal:
            return Bias.BEARISH, trend_strength
        return Bias.NEUTRAL, trend_strength

    def _is_trend_friendly(
        self,
        *,
        higher_bias: Bias,
        trend_strength: int,
        volatility_state: VolatilityState,
    ) -> bool:
        return higher_bias != Bias.NEUTRAL and volatility_state != VolatilityState.HIGH

    def _assess_setup(
        self,
        higher_bias: Bias,
        setup_ctx: PreparedTimeframe,
        setup_key: str,
        *,
        reference_ctx: PreparedTimeframe | None = None,
        current_price: float | None = None,
    ) -> dict[str, Any]:
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []
        reference_ctx = reference_ctx or setup_ctx
        divergence = setup_ctx.divergence_profile
        min_level = int(self.config.get("divergence", {}).get("min_level", 2))
        counter_trend_block_threshold = int(self.config.get("exhaustion", {}).get("counter_trend_block_threshold", 90))

        if higher_bias == Bias.NEUTRAL:
            return {
                "aligned": False,
                "execution_ready": False,
                "pullback_ready": False,
                "reversal_ready": True,
                "require_reversal_candle": False,
                "divergence_enabled": True,
                "require_divergence_gate": True,
                "require_free_space_gate": False,
                "divergence_ready": False,
                "divergence_level": 0,
                "opposing_divergence_level": 0,
                "free_space_ready": True,
                "free_space_r": None,
                "free_space_min_r": 0.0,
                "is_extended": False,
                "distance_to_value_atr": 0.0,
                "structure_ready": False,
                "score": -12,
                "score_note": f"{setup_key} 需要先看到有效 divergence，才谈 exhaustion reversal。",
                "reasons_for": reasons_for,
                "reasons_against": [f"{setup_key} 当前没有达到 L{min_level}+ 的单边 divergence。"],
                "risk_notes": risk_notes,
            }

        if higher_bias == Bias.BULLISH:
            divergence_level = int(divergence.get("bullish_level", 0))
            divergence_ready = bool(divergence.get("bullish_signal", False)) and divergence_level >= min_level
            opposing_divergence_level = int(divergence.get("bearish_level", 0))
            macro_blocked = (
                reference_ctx.model.trend_bias == Bias.BEARISH
                and int(reference_ctx.model.trend_score) >= counter_trend_block_threshold
            )
            if divergence_ready:
                reasons_for.append(
                    f"{setup_key} 出现 Bull divergence L{divergence_level}，先有 exhaustion 再谈 reversal。"
                )
            else:
                reasons_against.append(f"{setup_key} Bull divergence 还没达到 L{min_level}+。")
            if macro_blocked:
                reasons_against.append(f"4H 仍是过强空头环境，直接抄底风险太高。")
            else:
                reasons_for.append("4H 反向趋势没有强到禁止做反弹。")
        else:
            divergence_level = int(divergence.get("bearish_level", 0))
            divergence_ready = bool(divergence.get("bearish_signal", False)) and divergence_level >= min_level
            opposing_divergence_level = int(divergence.get("bullish_level", 0))
            macro_blocked = (
                reference_ctx.model.trend_bias == Bias.BULLISH
                and int(reference_ctx.model.trend_score) >= counter_trend_block_threshold
            )
            if divergence_ready:
                reasons_for.append(
                    f"{setup_key} 出现 Bear divergence L{divergence_level}，先有 exhaustion 再谈 reversal。"
                )
            else:
                reasons_against.append(f"{setup_key} Bear divergence 还没达到 L{min_level}+。")
            if macro_blocked:
                reasons_against.append(f"4H 仍是过强多头环境，直接摸顶风险太高。")
            else:
                reasons_for.append("4H 反向趋势没有强到禁止做回落。")

        execution_ready = divergence_ready and not macro_blocked
        score = 18 if execution_ready else 8 if divergence_ready else -10
        if macro_blocked:
            score -= 8

        return {
            "aligned": True,
            "execution_ready": execution_ready,
            "pullback_ready": execution_ready,
            "reversal_ready": True,
            "require_reversal_candle": False,
            "divergence_enabled": True,
            "require_divergence_gate": True,
            "require_free_space_gate": False,
            "divergence_ready": divergence_ready,
            "divergence_level": divergence_level,
            "opposing_divergence_level": opposing_divergence_level,
            "free_space_ready": True,
            "free_space_r": None,
            "free_space_min_r": 0.0,
            "is_extended": False,
            "distance_to_value_atr": 0.0,
            "structure_ready": True,
            "score": score,
            "score_note": f"{setup_key} 先要求 L{min_level}+ divergence，再检查 4H 是否反向过强。",
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
        }

    def _assess_trigger(
        self,
        higher_bias: Bias,
        trigger_ctx: PreparedTimeframe,
        trigger_key: str,
        *,
        trend_strength: int | None = None,
    ) -> dict[str, Any]:
        if higher_bias == Bias.NEUTRAL:
            return {
                "state": TriggerState.NONE,
                "score": -10,
                "score_note": f"{trigger_key} 当前没有 exhaustion reversal 方向。",
                "reasons_for": [],
                "reasons_against": [f"{trigger_key} 当前没有可执行的 divergence 方向。"],
                "risk_notes": [],
                "recent_low": None,
                "prior_low": None,
                "recent_high": None,
                "prior_high": None,
                "bullish_rejection": False,
                "bearish_rejection": False,
                "volume_contracting": False,
                "no_new_extreme": False,
                "regained_fast": False,
                "held_slow": False,
                "auxiliary_count": 0,
                "min_auxiliary_confirmations": 1,
            }

        latest = trigger_ctx.df.iloc[-1]
        ema21 = float(latest["ema_21"])
        close = float(latest["close"])
        candle_profile = trigger_ctx.candle_profile
        divergence = trigger_ctx.divergence_profile
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []

        if higher_bias == Bias.BULLISH:
            divergence_level = int(divergence.get("bullish_level", 0))
            divergence_ready = bool(divergence.get("bullish_signal", False))
            regained_fast = close >= ema21
            rejection = bool(candle_profile.get("has_bullish_reversal_candle", False))
            if divergence_ready:
                reasons_for.append(f"{trigger_key} Bull divergence L{divergence_level} 已成立。")
            if regained_fast:
                reasons_for.append(f"{trigger_key} 已重新站回 EMA21，上行反抽开始接管。")
            else:
                reasons_against.append(f"{trigger_key} 还没站回 EMA21，上行动能还不够。")
            if rejection:
                reasons_for.append(f"{trigger_key} 出现止跌 reversal candle。")
            else:
                reasons_against.append(f"{trigger_key} 缺少干净的止跌 reversal candle。")

            if divergence_ready and regained_fast:
                state = TriggerState.BULLISH_CONFIRMED
                score = 14
            elif divergence_ready and rejection:
                state = TriggerState.MIXED
                score = 1
                risk_notes.append(f"{trigger_key} 先有 rejection，但还没完成 EMA21 reclaim。")
            else:
                state = TriggerState.NONE
                score = -12
        else:
            divergence_level = int(divergence.get("bearish_level", 0))
            divergence_ready = bool(divergence.get("bearish_signal", False))
            regained_fast = close <= ema21
            rejection = bool(candle_profile.get("has_bearish_reversal_candle", False))
            if divergence_ready:
                reasons_for.append(f"{trigger_key} Bear divergence L{divergence_level} 已成立。")
            if regained_fast:
                reasons_for.append(f"{trigger_key} 已重新跌回 EMA21，下行动能开始接管。")
            else:
                reasons_against.append(f"{trigger_key} 还没跌回 EMA21，下行动能还不够。")
            if rejection:
                reasons_for.append(f"{trigger_key} 出现见顶 reversal candle。")
            else:
                reasons_against.append(f"{trigger_key} 缺少干净的见顶 reversal candle。")

            if divergence_ready and regained_fast:
                state = TriggerState.BEARISH_CONFIRMED
                score = 14
            elif divergence_ready and rejection:
                state = TriggerState.MIXED
                score = 1
                risk_notes.append(f"{trigger_key} 先有 rejection，但还没完成 EMA21 loss。")
            else:
                state = TriggerState.NONE
                score = -12

        return {
            "state": state,
            "score": score,
            "score_note": f"{trigger_key} 只看 divergence 后是否完成 EMA21 reclaim/loss。",
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
            "recent_low": round(float(latest["low"]), 4),
            "prior_low": round(float(latest["low"]), 4),
            "recent_high": round(float(latest["high"]), 4),
            "prior_high": round(float(latest["high"]), 4),
            "bullish_rejection": bool(candle_profile.get("has_bullish_reversal_candle", False)),
            "bearish_rejection": bool(candle_profile.get("has_bearish_reversal_candle", False)),
            "volume_contracting": bool(candle_profile.get("is_volume_contracting", False)),
            "no_new_extreme": True,
            "regained_fast": regained_fast,
            "held_slow": True,
            "auxiliary_count": int(regained_fast) + int(rejection),
            "min_auxiliary_confirmations": 1,
        }

    def _decide(
        self,
        *,
        higher_bias: Bias,
        trend_friendly: bool,
        setup_assessment: dict[str, Any],
        trigger_assessment: dict[str, Any],
        confidence: int,
    ) -> tuple[Action, Bias, RecommendedTiming]:
        threshold = int(self.config["confidence"]["action_threshold"])
        if higher_bias == Bias.BULLISH:
            if (
                trend_friendly
                and setup_assessment["pullback_ready"]
                and trigger_assessment["state"] == TriggerState.BULLISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.LONG, Bias.BULLISH, RecommendedTiming.NOW
            if trigger_assessment["state"] == TriggerState.MIXED:
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION
            return Action.WAIT, Bias.BULLISH, RecommendedTiming.SKIP
        if higher_bias == Bias.BEARISH:
            if (
                trend_friendly
                and setup_assessment["pullback_ready"]
                and trigger_assessment["state"] == TriggerState.BEARISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.SHORT, Bias.BEARISH, RecommendedTiming.NOW
            if trigger_assessment["state"] == TriggerState.MIXED:
                return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_CONFIRMATION
            return Action.WAIT, Bias.BEARISH, RecommendedTiming.SKIP
        return Action.WAIT, Bias.NEUTRAL, RecommendedTiming.SKIP

    def _build_trade_plan(
        self,
        *,
        action: Action,
        bias: Bias,
        setup_ctx: PreparedTimeframe,
        reference_ctx: PreparedTimeframe,
        current_price: float,
        setup_key: str,
        reference_key: str,
    ) -> dict[str, Any]:
        if action == Action.WAIT:
            if bias == Bias.BULLISH:
                return {
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": f"当前仍在等 {setup_key} 的 divergence 完成 EMA21 reclaim。",
                    "invalidation_price": None,
                }
            if bias == Bias.BEARISH:
                return {
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": f"当前仍在等 {setup_key} 的 divergence 完成 EMA21 loss。",
                    "invalidation_price": None,
                }
            return {
                "entry_zone": None,
                "stop_loss": None,
                "take_profit_hint": None,
                "invalidation": "当前没有可执行的 divergence exhaustion reversal。",
                "invalidation_price": None,
            }

        latest = setup_ctx.df.iloc[-1]
        atr = float(setup_ctx.model.atr14)
        atr_stop_buffer = float(self.config.get("exhaustion", {}).get("atr_stop_buffer", 0.18)) * atr
        minimum_risk = float(self.config.get("exhaustion", {}).get("minimum_r_multiple", 0.8)) * atr
        entry_pad = float(self.config.get("exhaustion", {}).get("entry_pad_atr", 0.08)) * atr

        if bias == Bias.BULLISH:
            anchor_low = float(setup_ctx.model.swing_low or latest["low"])
            stop_price = min(anchor_low, float(latest["low"])) - atr_stop_buffer
            risk = max(current_price - stop_price, minimum_risk)
            tp1 = current_price + risk
            tp2 = current_price + (2 * risk)
            return {
                "entry_zone": EntryZone(
                    low=round(current_price - entry_pad, 4),
                    high=round(current_price + entry_pad, 4),
                    basis=f"{setup_key} divergence reclaim 带",
                ),
                "stop_loss": StopLoss(
                    price=round(stop_price, 4),
                    basis=f"{setup_key} 最近低点下方加 ATR 缓冲",
                ),
                "take_profit_hint": TakeProfitHint(
                    tp1=round(tp1, 4),
                    tp2=round(tp2, 4),
                    basis="先看 1R 反抽，再看 2R mean-reversion extension",
                ),
                "invalidation": f"若 {setup_key} 再次跌破最近低点，则 bullish divergence 失败",
                "invalidation_price": round(min(anchor_low, float(latest['low'])), 4),
            }

        anchor_high = float(setup_ctx.model.swing_high or latest["high"])
        stop_price = max(anchor_high, float(latest["high"])) + atr_stop_buffer
        risk = max(stop_price - current_price, minimum_risk)
        tp1 = current_price - risk
        tp2 = current_price - (2 * risk)
        return {
            "entry_zone": EntryZone(
                low=round(current_price - entry_pad, 4),
                high=round(current_price + entry_pad, 4),
                basis=f"{setup_key} divergence loss 带",
            ),
            "stop_loss": StopLoss(
                price=round(stop_price, 4),
                basis=f"{setup_key} 最近高点上方加 ATR 缓冲",
            ),
            "take_profit_hint": TakeProfitHint(
                tp1=round(tp1, 4),
                tp2=round(tp2, 4),
                basis="先看 1R 回落，再看 2R mean-reversion extension",
            ),
            "invalidation": f"若 {setup_key} 再次突破最近高点，则 bearish divergence 失败",
            "invalidation_price": round(max(anchor_high, float(latest['high'])), 4),
        }

    def _build_summary(
        self,
        *,
        action: Action,
        recommended_timing: RecommendedTiming,
        higher_bias: Bias,
        setup_assessment: dict[str, Any],
        trigger_assessment: dict[str, Any],
        setup_key: str,
        trigger_key: str,
    ) -> str:
        if action == Action.LONG:
            return f"{trigger_key} 已出现 bullish divergence 并完成 EMA21 reclaim，因此按 exhaustion reversal 做多。"
        if action == Action.SHORT:
            return f"{trigger_key} 已出现 bearish divergence 并完成 EMA21 loss，因此按 exhaustion reversal 做空。"
        if higher_bias == Bias.BULLISH:
            return f"{trigger_key} 已有 bullish divergence 轮廓，但还没完成 EMA21 reclaim。"
        if higher_bias == Bias.BEARISH:
            return f"{trigger_key} 已有 bearish divergence 轮廓，但还没完成 EMA21 loss。"
        return "当前没有达到可执行的 divergence exhaustion reversal。"
