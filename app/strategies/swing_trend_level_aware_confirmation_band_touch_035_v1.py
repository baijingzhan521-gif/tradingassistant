from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_trend_level_aware_confirmation_v1 import (
    DEFAULT_CONFIG as SWING_TREND_LEVEL_AWARE_CONFIRMATION_V1_DEFAULT_CONFIG,
    SwingTrendLevelAwareConfirmationV1Strategy,
)


DEFAULT_CONFIG = deepcopy(SWING_TREND_LEVEL_AWARE_CONFIRMATION_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["level_confirmation"]["band_touch_proximity_atr"] = 0.35


class SwingTrendLevelAwareConfirmationBandTouch035V1Strategy(SwingTrendLevelAwareConfirmationV1Strategy):
    name = "swing_trend_level_aware_confirmation_band_touch_035_v1"
