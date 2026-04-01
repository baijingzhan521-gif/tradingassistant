from __future__ import annotations

from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_reversal_candle": False,
    },
    "trigger": {
        **COMMON_DEFAULT_CONFIG["trigger"],
        "bullish_require_regained_fast": True,
        "bullish_require_held_slow": True,
        "bullish_require_auxiliary": False,
        "min_auxiliary_confirmations": 0,
    },
    "confidence": {"action_threshold": 60},
    "window": {
        "higher_timeframes": ["1d", "4h"],
        "setup_timeframe": "1h",
        "trigger_timeframe": "1h",
        "reference_timeframe": "4h",
        "display_timeframes": ["1d", "4h", "1h"],
        "chart_timeframes": ["1d", "4h", "1h"],
        "trend_strength_threshold": 50,
        "bullish_trend_strength_threshold": 72,
        "bearish_trend_strength_threshold": 50,
    },
    "backtest": {"cooldown_bars_after_exit": 0},
}


class SwingImprovedV1Strategy(WindowedMTFStrategy):
    name = "swing_improved_v1"
    required_timeframes = ("1d", "4h", "1h")
