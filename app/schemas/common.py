from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    pass


class Action(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"


class Bias(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RecommendedTiming(StrEnum):
    NOW = "now"
    WAIT_PULLBACK = "wait_pullback"
    WAIT_CONFIRMATION = "wait_confirmation"
    SKIP = "skip"


class MarketType(StrEnum):
    PERPETUAL = "perpetual"


class SupportedTimeframe(StrEnum):
    DAY_1 = "1d"
    HOUR_4 = "4h"
    HOUR_1 = "1h"
    MIN_15 = "15m"
    MIN_3 = "3m"


class VolatilityState(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class EmaAlignment(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    MIXED = "mixed"


class StructureState(StrEnum):
    BULLISH = "higher_highs_higher_lows"
    BEARISH = "lower_highs_lower_lows"
    MIXED = "mixed"


class TriggerState(StrEnum):
    BULLISH_CONFIRMED = "bullish_confirmed"
    BEARISH_CONFIRMED = "bearish_confirmed"
    MIXED = "mixed"
    NONE = "none"
    NOT_APPLICABLE = "not_applicable"
