from __future__ import annotations

from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
)
from app.strategies.windowed_mtf import WindowedMTFStrategy


DEFAULT_CONFIG = {
    **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    "trigger": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG["trigger"],
        "bearish_require_regained_fast": False,
    },
}


class SwingTrendLongRegimeShortRelaxedTriggerV1Strategy(WindowedMTFStrategy):
    name = "swing_trend_long_regime_short_relaxed_trigger_v1"
    required_timeframes = ("1d", "4h", "1h")
