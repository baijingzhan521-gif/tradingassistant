from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_exhaustion_divergence_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceV1BTCStrategy,
)


DEFAULT_CONFIG = deepcopy(SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG)
DEFAULT_CONFIG["exhaustion"]["counter_trend_block_threshold"] = 80


class SwingExhaustionDivergenceCTBlock80V1BTCStrategy(SwingExhaustionDivergenceV1BTCStrategy):
    name = "swing_exhaustion_divergence_ct_block80_v1_btc"
