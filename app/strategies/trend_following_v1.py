from __future__ import annotations

from typing import Any, Optional

from app.schemas.common import Action, Bias, RecommendedTiming, StructureState, TriggerState
from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, PreparedTimeframe, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_reversal_candle": False,
    },
    "trigger": {
        **COMMON_DEFAULT_CONFIG["trigger"],
        "bullish_require_regained_fast": True,
        "bullish_require_held_slow": False,
        "bullish_require_auxiliary": False,
        "min_auxiliary_confirmations": 0,
    },
    "confidence": {"action_threshold": 55},
    "window": {
        "higher_timeframes": ["1d", "4h"],
        "setup_timeframe": "4h",
        "trigger_timeframe": "4h",
        "reference_timeframe": "1d",
        "display_timeframes": ["1d", "4h", "1h"],
        "chart_timeframes": ["1d", "4h", "1h"],
        "trend_strength_threshold": 40,
        "bullish_trend_strength_threshold": 40,
        "bearish_trend_strength_threshold": 40,
    },
}


class TrendFollowingV1Strategy(WindowedMTFStrategy):
    name = "trend_following_v1"
    required_timeframes = ("1d", "4h", "1h")

    # ------------------------------------------------------------------
    # Setup: Donchian-style breakout instead of pullback
    # ------------------------------------------------------------------
    def _assess_setup(
        self,
        higher_bias: Bias,
        setup_ctx: PreparedTimeframe,
        setup_key: str,
        *,
        reference_ctx: Optional[PreparedTimeframe] = None,
        current_price: Optional[float] = None,
    ) -> dict[str, Any]:
        current_price = float(current_price if current_price is not None else setup_ctx.model.close)
        score = 0

        if higher_bias == Bias.NEUTRAL:
            return {
                "aligned": False,
                "execution_ready": False,
                "pullback_ready": True,
                "reversal_ready": True,
                "require_reversal_candle": False,
                "require_free_space_gate": False,
                "free_space_ready": True,
                "is_extended": False,
                "score": -5,
                "score_note": "breakout setup assessment",
            }

        if higher_bias == Bias.BULLISH:
            price_above_ema200 = current_price > setup_ctx.model.ema200
            aligned = price_above_ema200
            if aligned:
                score += 10
            if setup_ctx.model.structure_state == StructureState.BULLISH:
                score += 5
        else:
            price_below_ema200 = current_price < setup_ctx.model.ema200
            aligned = price_below_ema200
            if aligned:
                score += 10
            if setup_ctx.model.structure_state == StructureState.BEARISH:
                score += 5

        if not aligned:
            score -= 5

        return {
            "aligned": aligned,
            "execution_ready": True,
            "pullback_ready": True,
            "reversal_ready": True,
            "require_reversal_candle": False,
            "require_free_space_gate": False,
            "free_space_ready": True,
            "is_extended": False,
            "score": score,
            "score_note": "breakout setup assessment",
        }

    # ------------------------------------------------------------------
    # Trigger: price vs EMA21 + new extreme check
    # ------------------------------------------------------------------
    def _assess_trigger(
        self,
        higher_bias: Bias,
        trigger_ctx: PreparedTimeframe,
        trigger_key: str,
        *,
        trend_strength: Optional[int] = None,
    ) -> dict[str, Any]:
        if higher_bias == Bias.NEUTRAL:
            return {
                "state": TriggerState.NONE,
                "score": -5,
                "score_note": "breakout trigger",
                "no_new_extreme": False,
                "regained_fast": False,
                "held_slow": True,
                "auxiliary_count": 0,
                "bullish_rejection": False,
                "bearish_rejection": False,
                "volume_contracting": False,
            }

        df = trigger_ctx.df
        latest = df.iloc[-1]
        close = float(latest["close"])
        ema21 = float(latest["ema_21"])
        recent_highs = df["high"].tail(3)
        recent_lows = df["low"].tail(3)
        recent_range = float(recent_highs.max()) - float(recent_lows.min())

        if higher_bias == Bias.BULLISH:
            above_ema21 = close > ema21
            near_high = recent_range > 0 and (close - float(recent_lows.min())) / recent_range >= 0.90
            confirmed = above_ema21 and near_high
            state = TriggerState.BULLISH_CONFIRMED if confirmed else TriggerState.NONE
            score = 12 if confirmed else -5
        else:
            below_ema21 = close < ema21
            near_low = recent_range > 0 and (float(recent_highs.max()) - close) / recent_range >= 0.90
            confirmed = below_ema21 and near_low
            state = TriggerState.BEARISH_CONFIRMED if confirmed else TriggerState.NONE
            score = 12 if confirmed else -5

        return {
            "state": state,
            "score": score,
            "score_note": "breakout trigger",
            "no_new_extreme": False,
            "regained_fast": close > ema21 if higher_bias == Bias.BULLISH else close < ema21,
            "held_slow": True,
            "auxiliary_count": 0,
            "bullish_rejection": False,
            "bearish_rejection": False,
            "volume_contracting": False,
        }

    # ------------------------------------------------------------------
    # Decide: simpler breakout logic
    # ------------------------------------------------------------------
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
                setup_assessment["aligned"]
                and trigger_assessment["state"] == TriggerState.BULLISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.LONG, Bias.BULLISH, RecommendedTiming.NOW
            return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION

        if higher_bias == Bias.BEARISH:
            if (
                setup_assessment["aligned"]
                and trigger_assessment["state"] == TriggerState.BEARISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.SHORT, Bias.BEARISH, RecommendedTiming.NOW
            return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_CONFIRMATION

        return Action.WAIT, Bias.NEUTRAL, RecommendedTiming.SKIP
