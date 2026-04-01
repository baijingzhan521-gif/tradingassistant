from __future__ import annotations

from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
)
from app.strategies.windowed_mtf import WindowedMTFStrategy


DEFAULT_CONFIG = {
    **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    "setup_confluence": {
        **SWING_LONG_REGIME_GATE_DEFAULT_CONFIG.get("setup_confluence", {}),
        "enabled": True,
        "min_hits": 2,
        "ema55_proximity_atr": 0.6,
        "pivot_proximity_atr": 0.9,
        "band_proximity_atr": 0.9,
        "max_spread_atr": 1.2,
    },
}


class SwingTrendConfluenceSetupV1Strategy(WindowedMTFStrategy):
    name = "swing_trend_confluence_setup_v1"
    required_timeframes = ("1d", "4h", "1h")
