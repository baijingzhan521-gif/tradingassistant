from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_breakout_v1_btc import (
    DEFAULT_CONFIG as SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG,
    SwingBreakoutV1BTCStrategy,
)


DEFAULT_CONFIG = deepcopy(SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG)
DEFAULT_CONFIG["breakout"]["setup_proximity_atr"] = 0.45


class SwingBreakoutSetupProximity045V1BTCStrategy(SwingBreakoutV1BTCStrategy):
    name = "swing_breakout_setup_proximity_045_v1_btc"
