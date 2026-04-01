from __future__ import annotations

from math import isfinite
from typing import Optional, Union


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def pct_distance(current: float, reference: float) -> float:
    if reference == 0:
        return 0.0
    return ((current - reference) / reference) * 100.0


def safe_float(value: Optional[Union[float, int]]) -> Optional[float]:
    if value is None:
        return None
    converted = float(value)
    if not isfinite(converted):
        return None
    return converted


def midpoint(low: float, high: float) -> float:
    return (low + high) / 2.0


def safe_ratio(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 when denominator is 0."""
    if denominator == 0:
        return 0.0
    return numerator / denominator
