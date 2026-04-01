from __future__ import annotations

from app.strategies.swing_trend_confluence_setup_v1 import (
    DEFAULT_CONFIG as SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG,
)
from app.strategies.windowed_mtf import WindowedMTFStrategy


DEFAULT_CONFIG = {
    **SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG,
    "setup": {
        **SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG["setup"],
        "require_structure_ready": True,
    },
}


class SwingTrendConfluenceStructureGateHardV1Strategy(WindowedMTFStrategy):
    name = "swing_trend_confluence_structure_gate_hard_v1"
    required_timeframes = ("1d", "4h", "1h")
