from __future__ import annotations

from copy import deepcopy

from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG as TREND_PULLBACK_V1_DEFAULT_CONFIG, TrendPullbackV1Strategy


DEFAULT_CONFIG = deepcopy(TREND_PULLBACK_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["execution"]["pullback_distance_atr"] = 0.85


class TrendPullbackPullback085V1Strategy(TrendPullbackV1Strategy):
    name = "trend_pullback_pullback_085_v1"
