from __future__ import annotations

import pandas as pd

from app.schemas.common import Bias, EmaAlignment, StructureState
from app.utils.math_utils import clamp


def determine_ema_alignment(ema21: float, ema55: float, ema100: float, ema200: float) -> EmaAlignment:
    if ema21 > ema55 > ema100 > ema200:
        return EmaAlignment.BULLISH
    if ema21 < ema55 < ema100 < ema200:
        return EmaAlignment.BEARISH
    return EmaAlignment.MIXED


def determine_trend_bias(close: float, ema200: float, alignment: EmaAlignment) -> Bias:
    if close > ema200 and alignment == EmaAlignment.BULLISH:
        return Bias.BULLISH
    if close < ema200 and alignment == EmaAlignment.BEARISH:
        return Bias.BEARISH
    return Bias.NEUTRAL


def classify_structure(swing_highs: list[float], swing_lows: list[float]) -> StructureState:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return StructureState.MIXED
    if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
        return StructureState.BULLISH
    if swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
        return StructureState.BEARISH
    return StructureState.MIXED


def compute_trend_strength(
    df: pd.DataFrame,
    *,
    close: float,
    ema21: float,
    ema55: float,
    ema100: float,
    ema200: float,
    alignment: EmaAlignment,
    bias: Bias,
    structure_state: StructureState,
) -> int:
    strength = 25.0
    if alignment != EmaAlignment.MIXED:
        strength += 20
    if bias != Bias.NEUTRAL:
        strength += 20

    if len(df) >= 5:
        ema21_slope = ema21 - float(df["ema_21"].iloc[-5])
        ema55_slope = ema55 - float(df["ema_55"].iloc[-5])
    else:
        ema21_slope = 0.0
        ema55_slope = 0.0

    if alignment == EmaAlignment.BULLISH and ema21_slope > 0 and ema55_slope > 0:
        strength += 15
    elif alignment == EmaAlignment.BEARISH and ema21_slope < 0 and ema55_slope < 0:
        strength += 15

    if structure_state == StructureState.BULLISH and bias == Bias.BULLISH:
        strength += 10
    elif structure_state == StructureState.BEARISH and bias == Bias.BEARISH:
        strength += 10

    ema_band = abs(ema21 - ema200) / close if close else 0.0
    strength += min(10.0, ema_band * 100)
    return int(clamp(round(strength), 0, 100))
