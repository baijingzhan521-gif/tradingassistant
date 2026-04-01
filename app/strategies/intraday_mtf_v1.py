from __future__ import annotations

from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_reversal_candle": True,
    },
    "window": {
        "higher_timeframes": ["1h"],
        "setup_timeframe": "15m",
        "trigger_timeframe": "3m",
        "reference_timeframe": "1h",
        "display_timeframes": ["1h", "15m", "3m"],
        "chart_timeframes": ["1h", "15m", "3m"],
        "trend_strength_threshold": 55,
    },
}


class IntradayMTFV1Strategy(WindowedMTFStrategy):
    name = "intraday_mtf_v1"
    required_timeframes = ("1h", "15m", "3m")
