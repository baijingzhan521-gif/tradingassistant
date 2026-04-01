from __future__ import annotations

from copy import deepcopy

from app.strategies.swing_trend_confluence_setup_v1 import (
    DEFAULT_CONFIG as SWING_TREND_CONFLUENCE_SETUP_V1_DEFAULT_CONFIG,
    SwingTrendConfluenceSetupV1Strategy,
)


DEFAULT_CONFIG = deepcopy(SWING_TREND_CONFLUENCE_SETUP_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["setup_confluence"]["min_hits"] = 3


class SwingTrendConfluenceMinHits3V1Strategy(SwingTrendConfluenceSetupV1Strategy):
    name = "swing_trend_confluence_min_hits_3_v1"
