from __future__ import annotations

from typing import Any, Optional

from app.schemas.common import Action, Bias, RecommendedTiming, TriggerState, VolatilityState
from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, PreparedTimeframe, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_reversal_candle": True,
    },
    "trigger": {
        **COMMON_DEFAULT_CONFIG["trigger"],
        "bullish_require_regained_fast": False,
        "bullish_require_held_slow": False,
        "bullish_require_auxiliary": False,
        "min_auxiliary_confirmations": 0,
    },
    "confidence": {"action_threshold": 58},
    "window": {
        "higher_timeframes": ["1d"],
        "setup_timeframe": "1h",
        "trigger_timeframe": "1h",
        "reference_timeframe": "4h",
        "display_timeframes": ["1d", "4h", "1h"],
        "chart_timeframes": ["1d", "4h", "1h"],
        "trend_strength_threshold": 0,
        "bullish_trend_strength_threshold": 0,
        "bearish_trend_strength_threshold": 0,
    },
    "backtest": {"cooldown_bars_after_exit": 1},
    "mean_reversion": {
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "bb_pctb_low": 0.05,
        "bb_pctb_high": 0.95,
    },
}


class MeanReversionV1Strategy(WindowedMTFStrategy):
    name = "mean_reversion_v1"
    required_timeframes = ("1d", "4h", "1h")

    # ------------------------------------------------------------------
    # Always trend-friendly – mean reversion trades in any regime
    # ------------------------------------------------------------------
    def _is_trend_friendly(
        self,
        *,
        higher_bias: Bias,
        trend_strength: int,
        volatility_state: VolatilityState,
    ) -> bool:
        return True

    # ------------------------------------------------------------------
    # Setup: Bollinger Band extreme detection
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
        mr_config = dict(self.config.get("mean_reversion", {}))
        bb_pctb_low = float(mr_config.get("bb_pctb_low", 0.05))
        bb_pctb_high = float(mr_config.get("bb_pctb_high", 0.95))
        band_position = setup_ctx.band_position
        candle_profile = setup_ctx.candle_profile
        score = 0

        if higher_bias == Bias.NEUTRAL:
            return {
                "aligned": False,
                "execution_ready": False,
                "pullback_ready": True,
                "reversal_ready": False,
                "require_reversal_candle": True,
                "require_free_space_gate": False,
                "free_space_ready": True,
                "is_extended": False,
                "score": -5,
                "score_note": "mean reversion setup assessment",
            }

        if higher_bias == Bias.BULLISH:
            at_extreme = band_position < bb_pctb_low
            near_extreme = band_position < min(bb_pctb_low * 3, 0.20)
            reversal_ready = bool(candle_profile.get("has_bullish_reversal_candle", False))
        else:
            at_extreme = band_position > bb_pctb_high
            near_extreme = band_position > max(1.0 - (1.0 - bb_pctb_high) * 3, 0.80)
            reversal_ready = bool(candle_profile.get("has_bearish_reversal_candle", False))

        if at_extreme:
            score += 10
        elif near_extreme:
            score += 5
        else:
            score -= 10

        aligned = at_extreme or near_extreme

        return {
            "aligned": aligned,
            "execution_ready": True,
            "pullback_ready": True,
            "reversal_ready": reversal_ready,
            "require_reversal_candle": True,
            "require_free_space_gate": False,
            "free_space_ready": True,
            "is_extended": False,
            "score": score,
            "score_note": "mean reversion setup assessment",
        }

    # ------------------------------------------------------------------
    # Trigger: reversal candle at band extreme
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
                "score_note": "mean reversion trigger",
                "no_new_extreme": False,
                "regained_fast": False,
                "held_slow": True,
                "auxiliary_count": 0,
                "bullish_rejection": False,
                "bearish_rejection": False,
                "volume_contracting": False,
            }

        mr_config = dict(self.config.get("mean_reversion", {}))
        bb_pctb_low = float(mr_config.get("bb_pctb_low", 0.05))
        bb_pctb_high = float(mr_config.get("bb_pctb_high", 0.95))
        band_position = trigger_ctx.band_position
        candle_profile = trigger_ctx.candle_profile

        if higher_bias == Bias.BULLISH:
            at_band_extreme = band_position < min(bb_pctb_low * 3, 0.20)
            has_reversal = bool(candle_profile.get("has_bullish_reversal_candle", False))
            confirmed = at_band_extreme and has_reversal
            state = TriggerState.BULLISH_CONFIRMED if confirmed else TriggerState.NONE
        else:
            at_band_extreme = band_position > max(1.0 - (1.0 - bb_pctb_high) * 3, 0.80)
            has_reversal = bool(candle_profile.get("has_bearish_reversal_candle", False))
            confirmed = at_band_extreme and has_reversal
            state = TriggerState.BEARISH_CONFIRMED if confirmed else TriggerState.NONE

        score = 12 if confirmed else -5

        return {
            "state": state,
            "score": score,
            "score_note": "mean reversion trigger",
            "no_new_extreme": False,
            "regained_fast": False,
            "held_slow": True,
            "auxiliary_count": 0,
            "bullish_rejection": candle_profile.get("has_bullish_rejection", False),
            "bearish_rejection": candle_profile.get("has_bearish_rejection", False),
            "volume_contracting": candle_profile.get("is_volume_contracting", False),
        }

    # ------------------------------------------------------------------
    # Decide: ignores higher_bias direction – mean reversion is
    # counter-trend by nature
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
        reversal_ready = bool(setup_assessment.get("reversal_ready", False))

        if higher_bias == Bias.BULLISH:
            if (
                setup_assessment["aligned"]
                and reversal_ready
                and trigger_assessment["state"] == TriggerState.BULLISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.LONG, Bias.BULLISH, RecommendedTiming.NOW
            return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION

        if higher_bias == Bias.BEARISH:
            if (
                setup_assessment["aligned"]
                and reversal_ready
                and trigger_assessment["state"] == TriggerState.BEARISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.SHORT, Bias.BEARISH, RecommendedTiming.NOW
            return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_CONFIRMATION

        return Action.WAIT, Bias.NEUTRAL, RecommendedTiming.SKIP
