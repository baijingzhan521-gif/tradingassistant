from __future__ import annotations

from app.schemas.common import Bias
from app.strategies.swing_exhaustion_divergence_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceV1BTCStrategy,
)
from app.strategies.windowed_mtf import PreparedTimeframe


DEFAULT_CONFIG = SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG


class SwingExhaustionDivergenceShortOnlyV1BTCStrategy(SwingExhaustionDivergenceV1BTCStrategy):
    name = "swing_exhaustion_divergence_short_only_v1_btc"

    def _derive_higher_timeframe_bias(
        self, prepared: dict[str, PreparedTimeframe]
    ) -> tuple[Bias, int]:
        bias, trend_strength = super()._derive_higher_timeframe_bias(prepared)
        if bias == Bias.BULLISH:
            return Bias.NEUTRAL, trend_strength
        return bias, trend_strength
