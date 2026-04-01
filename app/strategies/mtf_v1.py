from __future__ import annotations

from app.strategies.intraday_mtf_v1 import (
    DEFAULT_CONFIG as INTRADAY_DEFAULT_CONFIG,
    IntradayMTFV1Strategy as IntradayMtfV1Strategy,
)
from app.strategies.swing_trend_v1 import (
    DEFAULT_CONFIG as SWING_DEFAULT_CONFIG,
    SwingTrendV1Strategy,
)
from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG as TREND_DEFAULT_CONFIG, TrendPullbackV1Strategy


DEFAULT_CONFIG = TREND_DEFAULT_CONFIG

__all__ = [
    "DEFAULT_CONFIG",
    "TrendPullbackV1Strategy",
    "SwingTrendV1Strategy",
    "IntradayMtfV1Strategy",
    "SWING_DEFAULT_CONFIG",
    "INTRADAY_DEFAULT_CONFIG",
]
