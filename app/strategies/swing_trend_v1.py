from __future__ import annotations

from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_reversal_candle": True,
    },
    "window": {
        "higher_timeframes": ["1d", "4h"],
        "setup_timeframe": "1h",
        "trigger_timeframe": "1h",
        "reference_timeframe": "4h",
        "display_timeframes": ["1d", "4h", "1h"],
        "chart_timeframes": ["1d", "4h", "1h"],
        "trend_strength_threshold": 60,
    },
}


class SwingTrendV1Strategy(WindowedMTFStrategy):
    name = "swing_trend_v1"
    required_timeframes = ("1d", "4h", "1h")
