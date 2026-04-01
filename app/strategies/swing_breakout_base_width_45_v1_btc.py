from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_breakout_v1_btc import (
    DEFAULT_CONFIG as SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG,
    SwingBreakoutV1BTCStrategy,
)


DEFAULT_CONFIG = deepcopy(SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG)
DEFAULT_CONFIG["breakout"]["base_max_width_atr"] = 4.5


class SwingBreakoutBaseWidth45V1BTCStrategy(SwingBreakoutV1BTCStrategy):
    name = "swing_breakout_base_width_45_v1_btc"
