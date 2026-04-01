from __future__ import annotations

from copy import deepcopy

from app.schemas.common import Bias, TriggerState
from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    SwingTrendLongRegimeGateV1Strategy,
)


def build_entry_attribution_config(
    *,
    include_reversal: bool,
    include_regained_fast: bool,
    include_held_slow: bool,
    include_auxiliary: bool,
) -> dict:
    config = deepcopy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    config["setup"]["require_reversal_candle"] = include_reversal
    config["trigger"]["bullish_require_regained_fast"] = include_regained_fast
    config["trigger"]["bearish_require_regained_fast"] = include_regained_fast
    config["trigger"]["bullish_require_held_slow"] = include_held_slow
    config["trigger"]["bearish_require_held_slow"] = include_held_slow
    config["trigger"]["bullish_require_auxiliary"] = include_auxiliary
    config["trigger"]["bearish_require_auxiliary"] = include_auxiliary
    return config


class SwingTrendEntryAttributionStrategy(SwingTrendLongRegimeGateV1Strategy):
    required_timeframes = ("1d", "4h", "1h")

    def __init__(self, config: dict, *, profile_name: str) -> None:
        super().__init__(config)
        self.name = profile_name

    def _assess_trigger(
        self,
        higher_bias: Bias,
        trigger_ctx,
        trigger_key: str,
        *,
        trend_strength: int | None = None,
    ) -> dict:
        assessment = super()._assess_trigger(
            higher_bias,
            trigger_ctx,
            trigger_key,
            trend_strength=trend_strength,
        )
        if higher_bias == Bias.NEUTRAL:
            return assessment

        requirements = self._resolve_trigger_requirements(higher_bias, trend_strength)
        if any(
            (
                requirements["require_regained_fast"],
                requirements["require_held_slow"],
                requirements["require_auxiliary"],
            )
        ):
            return assessment

        assessment["state"] = (
            TriggerState.BULLISH_CONFIRMED if higher_bias == Bias.BULLISH else TriggerState.BEARISH_CONFIRMED
        )
        assessment["score"] = 0
        assessment["score_note"] = f"{trigger_key} 不启用 current entry trigger 细节过滤"
        assessment["reasons_against"] = []
        assessment["risk_notes"] = []
        return assessment
