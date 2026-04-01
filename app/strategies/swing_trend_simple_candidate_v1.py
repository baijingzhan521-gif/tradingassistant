from __future__ import annotations

from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
)
from app.strategies.windowed_mtf import WindowedMTFStrategy


DEFAULT_CONFIG = {
    **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    "setup": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG["setup"],
        "require_reversal_candle": False,
    },
    "trigger": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG["trigger"],
        "bullish_require_held_slow": False,
        "bearish_require_held_slow": False,
        "bullish_require_auxiliary": False,
        "bearish_require_auxiliary": False,
    },
}


class SwingTrendSimpleCandidateV1Strategy(WindowedMTFStrategy):
    name = "swing_trend_simple_candidate_v1"
    required_timeframes = ("1d", "4h", "1h")
