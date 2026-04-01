from __future__ import annotations

from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "micro": {
        **COMMON_DEFAULT_CONFIG["micro"],
        "confirmation_lookback": 24,
    },
    "execution": {
        **COMMON_DEFAULT_CONFIG["execution"],
        "pullback_distance_atr": 0.85,
        "support_proximity_atr": 0.55,
        "resistance_proximity_atr": 0.55,
    },
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_structure_ready": True,
        "require_reversal_candle": True,
    },
    "trigger": {
        **COMMON_DEFAULT_CONFIG["trigger"],
        "min_auxiliary_confirmations": 2,
        "mixed_score": -6,
        "none_score": -14,
    },
    "backtest": {
        **COMMON_DEFAULT_CONFIG["backtest"],
        "cooldown_bars_after_exit": 15,
    },
    "window": {
        "higher_timeframes": ["1h"],
        "setup_timeframe": "15m",
        "trigger_timeframe": "3m",
        "reference_timeframe": "1h",
        "display_timeframes": ["1h", "15m", "3m"],
        "chart_timeframes": ["1h", "15m", "3m"],
        "trend_strength_threshold": 65,
    },
}


class IntradayMTFV2Strategy(WindowedMTFStrategy):
    name = "intraday_mtf_v2"
    required_timeframes = ("1h", "15m", "3m")
