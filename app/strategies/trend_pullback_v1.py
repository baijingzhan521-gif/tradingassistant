from __future__ import annotations

from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "window": {
        "higher_timeframes": ["1d", "4h"],
        "setup_timeframe": "1h",
        "trigger_timeframe": "15m",
        "reference_timeframe": "4h",
        "display_timeframes": ["1d", "4h", "1h", "15m", "3m"],
        "chart_timeframes": ["1d", "4h", "1h", "15m", "3m"],
        "trend_strength_threshold": 60,
    },
}


class TrendPullbackV1Strategy(WindowedMTFStrategy):
    name = "trend_pullback_v1"
    required_timeframes = ("1d", "4h", "1h", "15m")
