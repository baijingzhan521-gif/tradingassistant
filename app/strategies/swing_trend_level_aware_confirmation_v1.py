from __future__ import annotations

from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
)
from app.strategies.windowed_mtf import WindowedMTFStrategy


DEFAULT_CONFIG = {
    **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    "level_confirmation": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG.get("level_confirmation", {}),
        "enabled": True,
        "min_hits": 1,
        "ema55_touch_proximity_atr": 0.35,
        "pivot_touch_proximity_atr": 0.5,
        "band_touch_proximity_atr": 0.5,
    },
}


class SwingTrendLevelAwareConfirmationV1Strategy(WindowedMTFStrategy):
    name = "swing_trend_level_aware_confirmation_v1"
    required_timeframes = ("1d", "4h", "1h")
