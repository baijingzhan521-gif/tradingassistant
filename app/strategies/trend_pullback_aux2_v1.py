from __future__ import annotations

from copy import deepcopy

from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG as TREND_PULLBACK_V1_DEFAULT_CONFIG, TrendPullbackV1Strategy


DEFAULT_CONFIG = deepcopy(TREND_PULLBACK_V1_DEFAULT_CONFIG)
DEFAULT_CONFIG["trigger"]["min_auxiliary_confirmations"] = 2


class TrendPullbackAux2V1Strategy(TrendPullbackV1Strategy):
    name = "trend_pullback_aux2_v1"
