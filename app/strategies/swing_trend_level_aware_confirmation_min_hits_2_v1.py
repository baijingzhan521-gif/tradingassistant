from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_trend_level_aware_confirmation_v1 import (
    DEFAULT_CONFIG as SWING_TREND_LEVEL_AWARE_CONFIRMATION_V1_DEFAULT_CONFIG,
    SwingTrendLevelAwareConfirmationV1Strategy,
)


DEFAULT_CONFIG = deepcopy(SWING_TREND_LEVEL_AWARE_CONFIRMATION_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["level_confirmation"]["min_hits"] = 2


class SwingTrendLevelAwareConfirmationMinHits2V1Strategy(SwingTrendLevelAwareConfirmationV1Strategy):
    name = "swing_trend_level_aware_confirmation_min_hits_2_v1"
