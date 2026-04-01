from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_trend_confluence_setup_v1 import (
    DEFAULT_CONFIG as SWING_TREND_CONFLUENCE_SETUP_V1_DEFAULT_CONFIG,
    SwingTrendConfluenceSetupV1Strategy,
)


DEFAULT_CONFIG = deepcopy(SWING_TREND_CONFLUENCE_SETUP_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["setup_confluence"]["max_spread_atr"] = 1.0


class SwingTrendConfluenceMaxSpread10V1Strategy(SwingTrendConfluenceSetupV1Strategy):
    name = "swing_trend_confluence_max_spread_10_v1"
