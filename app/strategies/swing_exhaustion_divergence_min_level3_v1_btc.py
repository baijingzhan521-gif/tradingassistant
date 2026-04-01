from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_exhaustion_divergence_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceV1BTCStrategy,
)


DEFAULT_CONFIG = deepcopy(SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG)
DEFAULT_CONFIG["divergence"]["min_level"] = 3


class SwingExhaustionDivergenceMinLevel3V1BTCStrategy(SwingExhaustionDivergenceV1BTCStrategy):
    name = "swing_exhaustion_divergence_min_level3_v1_btc"
