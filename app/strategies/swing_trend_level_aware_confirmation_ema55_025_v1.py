from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_trend_level_aware_confirmation_v1 import (
    DEFAULT_CONFIG as SWING_TREND_LEVEL_AWARE_CONFIRMATION_V1_DEFAULT_CONFIG,
    SwingTrendLevelAwareConfirmationV1Strategy,
)


DEFAULT_CONFIG = deepcopy(SWING_TREND_LEVEL_AWARE_CONFIRMATION_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["level_confirmation"]["ema55_touch_proximity_atr"] = 0.25


class SwingTrendLevelAwareConfirmationEma55025V1Strategy(SwingTrendLevelAwareConfirmationV1Strategy):
    name = "swing_trend_level_aware_confirmation_ema55_025_v1"
