from __future__ import annotations

from copy import deepcopy

from app.strategies.intraday_mtf_v2 import DEFAULT_CONFIG as INTRADAY_MTF_V2_DEFAULT_CONFIG, IntradayMTFV2Strategy


DEFAULT_CONFIG = deepcopy(INTRADAY_MTF_V2_DEFAULT_CONFIG)
DEFAULT_CONFIG["window"]["trend_strength_threshold"] = 70


class IntradayMTFV2Trend70V1Strategy(IntradayMTFV2Strategy):
    name = "intraday_mtf_v2_trend70_v1"
