from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_trend_divergence_v1 import (
    DEFAULT_CONFIG as SWING_TREND_DIVERGENCE_V1_DEFAULT_CONFIG,
    SwingTrendDivergenceV1Strategy,
)


DEFAULT_CONFIG = deepcopy(SWING_TREND_DIVERGENCE_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["divergence"]["min_level"] = 3


class SwingTrendDivergenceMinLevel3V1Strategy(SwingTrendDivergenceV1Strategy):
    name = "swing_trend_divergence_min_level3_v1"
