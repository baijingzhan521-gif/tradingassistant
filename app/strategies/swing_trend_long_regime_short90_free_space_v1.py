from __future__ import annotations

from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
)
from app.strategies.windowed_mtf import WindowedMTFStrategy


DEFAULT_CONFIG = {
    **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    "trigger": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG["trigger"],
        "bearish_relax_regained_fast_at_trend_strength": 90,
    },
    "free_space": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG.get("free_space", {}),
        "enabled": True,
        "long_min_r": 1.0,
        "short_min_r": 0.0,
    },
}


class SwingTrendLongRegimeShort90FreeSpaceV1Strategy(WindowedMTFStrategy):
    name = "swing_trend_long_regime_short90_free_space_v1"
    required_timeframes = ("1d", "4h", "1h")
