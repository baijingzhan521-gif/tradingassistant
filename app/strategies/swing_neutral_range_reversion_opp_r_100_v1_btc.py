from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_neutral_range_reversion_v1_btc import (
    DEFAULT_CONFIG as SWING_NEUTRAL_RANGE_REVERSION_V1_BTC_DEFAULT_CONFIG,
    SwingNeutralRangeReversionV1BTCStrategy,
)


DEFAULT_CONFIG = deepcopy(SWING_NEUTRAL_RANGE_REVERSION_V1_BTC_DEFAULT_CONFIG)
DEFAULT_CONFIG["neutral_range"]["minimum_opposite_edge_r"] = 1.00


class SwingNeutralRangeReversionOppR100V1BTCStrategy(SwingNeutralRangeReversionV1BTCStrategy):
    name = "swing_neutral_range_reversion_opp_r_100_v1_btc"
