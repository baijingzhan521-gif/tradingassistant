from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_range_failure_v1_btc import (
    DEFAULT_CONFIG as SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG,
    SwingRangeFailureV1BTCStrategy,
)


DEFAULT_CONFIG = deepcopy(SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG)
DEFAULT_CONFIG["range_failure"]["max_width_atr"] = 4.5


class SwingRangeFailureMaxWidth45V1BTCStrategy(SwingRangeFailureV1BTCStrategy):
    name = "swing_range_failure_max_width_45_v1_btc"
