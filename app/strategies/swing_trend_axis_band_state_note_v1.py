from __future__ import annotations

from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
)
from app.strategies.windowed_mtf import WindowedMTFStrategy


DEFAULT_CONFIG = {
    **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    "state_note": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG.get("state_note", {}),
        "enabled": True,
        "bullish_axis_threshold": 2.0,
        "bullish_band_threshold": 0.9,
        "bullish_extreme_axis_threshold": 3.0,
        "bullish_extreme_band_threshold": 1.0,
        "bearish_axis_threshold": 2.0,
        "bearish_band_threshold": 0.1,
        "bearish_extreme_axis_threshold": 3.0,
        "bearish_extreme_band_threshold": 0.0,
    },
}


class SwingTrendAxisBandStateNoteV1Strategy(WindowedMTFStrategy):
    name = "swing_trend_axis_band_state_note_v1"
    required_timeframes = ("1d", "4h", "1h")
