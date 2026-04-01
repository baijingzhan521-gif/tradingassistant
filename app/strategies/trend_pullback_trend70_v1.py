from __future__ import annotations

from copy import deepcopy

from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG as TREND_PULLBACK_V1_DEFAULT_CONFIG, TrendPullbackV1Strategy


DEFAULT_CONFIG = deepcopy(TREND_PULLBACK_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["window"]["trend_strength_threshold"] = 70


class TrendPullbackTrend70V1Strategy(TrendPullbackV1Strategy):
    name = "trend_pullback_trend70_v1"
