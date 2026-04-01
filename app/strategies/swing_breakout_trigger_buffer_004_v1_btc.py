from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_breakout_v1_btc import (
    DEFAULT_CONFIG as SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG,
    SwingBreakoutV1BTCStrategy,
)


DEFAULT_CONFIG = deepcopy(SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG)
DEFAULT_CONFIG["breakout"]["trigger_breakout_buffer_atr"] = 0.04


class SwingBreakoutTriggerBuffer004V1BTCStrategy(SwingBreakoutV1BTCStrategy):
    name = "swing_breakout_trigger_buffer_004_v1_btc"
