from __future__ import annotations

from app.schemas.common import Action, Bias, RecommendedTiming, VolatilityState
from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    SwingTrendLongRegimeGateV1Strategy,
)


SIMPLE_ENTRY_CONFIG = {
    **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    "setup": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG["setup"],
        "require_reversal_candle": False,
    },
    "trigger": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG["trigger"],
        "bullish_require_regained_fast": False,
        "bearish_require_regained_fast": False,
        "bullish_require_held_slow": False,
        "bearish_require_held_slow": False,
        "bullish_require_auxiliary": False,
        "bearish_require_auxiliary": False,
    },
}

NO_GATE_CURRENT_ENTRY_DEFAULT_CONFIG = {
    **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
}

GATE_SIMPLE_ENTRY_DEFAULT_CONFIG = {
    **SIMPLE_ENTRY_CONFIG,
}

NO_GATE_SIMPLE_ENTRY_DEFAULT_CONFIG = {
    **SIMPLE_ENTRY_CONFIG,
}


class _SwingTrendGateEntryMatrixBaseStrategy(SwingTrendLongRegimeGateV1Strategy):
    enforce_regime_gate = True
    use_current_entry = True

    def _is_trend_friendly(
        self,
        *,
        higher_bias: Bias,
        trend_strength: int,
        volatility_state: VolatilityState,
    ) -> bool:
        if self.enforce_regime_gate:
            return super()._is_trend_friendly(
                higher_bias=higher_bias,
                trend_strength=trend_strength,
                volatility_state=volatility_state,
            )
        if higher_bias == Bias.NEUTRAL or volatility_state == VolatilityState.HIGH:
            return False
        return True

    def _decide(
        self,
        *,
        higher_bias: Bias,
        trend_friendly: bool,
        setup_assessment: dict[str, object],
        trigger_assessment: dict[str, object],
        confidence: int,
    ) -> tuple[Action, Bias, RecommendedTiming]:
        if self.use_current_entry:
            return super()._decide(
                higher_bias=higher_bias,
                trend_friendly=trend_friendly,
                setup_assessment=setup_assessment,
                trigger_assessment=trigger_assessment,
                confidence=confidence,
            )

        if higher_bias == Bias.BULLISH:
            if trend_friendly and bool(setup_assessment["aligned"]) and bool(setup_assessment["pullback_ready"]):
                return Action.LONG, Bias.BULLISH, RecommendedTiming.NOW
            if bool(setup_assessment["is_extended"]):
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_PULLBACK
            if bool(setup_assessment["aligned"]):
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION
            return Action.WAIT, Bias.BULLISH, RecommendedTiming.SKIP

        if higher_bias == Bias.BEARISH:
            if trend_friendly and bool(setup_assessment["aligned"]) and bool(setup_assessment["pullback_ready"]):
                return Action.SHORT, Bias.BEARISH, RecommendedTiming.NOW
            if bool(setup_assessment["is_extended"]):
                return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_PULLBACK
            if bool(setup_assessment["aligned"]):
                return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_CONFIRMATION
            return Action.WAIT, Bias.BEARISH, RecommendedTiming.SKIP

        return Action.WAIT, Bias.NEUTRAL, RecommendedTiming.SKIP


class SwingTrendMatrixNoGateCurrentEntryV1Strategy(_SwingTrendGateEntryMatrixBaseStrategy):
    name = "swing_trend_matrix_no_gate_current_entry_v1"
    required_timeframes = ("1d", "4h", "1h")
    enforce_regime_gate = False
    use_current_entry = True


class SwingTrendMatrixGateSimpleEntryV1Strategy(_SwingTrendGateEntryMatrixBaseStrategy):
    name = "swing_trend_matrix_gate_simple_entry_v1"
    required_timeframes = ("1d", "4h", "1h")
    enforce_regime_gate = True
    use_current_entry = False


class SwingTrendMatrixNoGateSimpleEntryV1Strategy(_SwingTrendGateEntryMatrixBaseStrategy):
    name = "swing_trend_matrix_no_gate_simple_entry_v1"
    required_timeframes = ("1d", "4h", "1h")
    enforce_regime_gate = False
    use_current_entry = False
