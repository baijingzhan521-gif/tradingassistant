from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_range_failure_v1_btc import (
    DEFAULT_CONFIG as SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG,
    SwingRangeFailureV1BTCStrategy,
)


DEFAULT_CONFIG = deepcopy(SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG)
DEFAULT_CONFIG["range_failure"]["edge_proximity_atr"] = 0.35


class SwingRangeFailureEdge035V1BTCStrategy(SwingRangeFailureV1BTCStrategy):
    name = "swing_range_failure_edge_035_v1_btc"
