from __future__ import annotations

from typing import Annotated, List, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import get_settings
from app.schemas.common import MarketType, SupportedTimeframe
from app.utils.timeframes import (
    MAINLINE_STRATEGY_PROFILE,
    SUPPORTED_STRATEGY_PROFILES,
    WORKSPACE_BATCH_PROFILES,
    normalize_timeframes,
    validate_required_timeframes,
)


settings = get_settings()


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    symbol: Annotated[str, Field(description="Unified ccxt symbol, e.g. ETH/USDT:USDT")]
    market_type: Annotated[MarketType, Field(default=MarketType(settings.default_market_type))]
    exchange: Annotated[str, Field(default=settings.default_exchange)]
    timeframes: Annotated[
        list[SupportedTimeframe],
        Field(
            description="Decision timeframes requested by the caller. The backend may auto-fetch supplemental 3m data for charts and micro observation.",
            default_factory=lambda: [
                SupportedTimeframe.DAY_1,
                SupportedTimeframe.HOUR_4,
                SupportedTimeframe.HOUR_1,
            ]
        ),
    ]
    strategy_profile: Annotated[str, Field(default=MAINLINE_STRATEGY_PROFILE)]
    lookback: Annotated[int, Field(default=settings.default_lookback, ge=220, le=1000)]

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("exchange")
    @classmethod
    def normalize_exchange(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("strategy_profile")
    @classmethod
    def normalize_strategy_profile(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("timeframes", mode="before")
    @classmethod
    def normalize_timeframe_values(
        cls, value: Union[List[str], List[SupportedTimeframe]]
    ) -> list[str]:
        normalized = [item.value if hasattr(item, "value") else str(item) for item in value]
        return normalize_timeframes(normalized)

    @model_validator(mode="after")
    def validate_strategy_contract(self) -> "AnalyzeRequest":
        normalized_timeframes = [
            item.value if hasattr(item, "value") else str(item) for item in self.timeframes
        ]
        if self.strategy_profile not in SUPPORTED_STRATEGY_PROFILES:
            raise ValueError(f"Unsupported strategy profile: {self.strategy_profile}")
        try:
            validate_required_timeframes(self.strategy_profile, normalized_timeframes)
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        return self


class WorkspaceBatchAnalyzeRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    symbol: Annotated[str, Field(description="Unified ccxt symbol, e.g. ETH/USDT:USDT")]
    market_type: Annotated[MarketType, Field(default=MarketType(settings.default_market_type))]
    exchange: Annotated[str, Field(default=settings.default_exchange)]
    strategy_profiles: Annotated[
        list[str],
        Field(
            default_factory=lambda: list(WORKSPACE_BATCH_PROFILES),
            description="Profiles to run in one workspace batch call.",
        ),
    ]
    lookback: Annotated[int, Field(default=settings.default_lookback, ge=220, le=1000)]

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("exchange")
    @classmethod
    def normalize_exchange(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("strategy_profiles", mode="before")
    @classmethod
    def normalize_strategy_profiles(cls, value: List[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value]
        unique = list(dict.fromkeys(normalized))
        invalid = [item for item in unique if item not in SUPPORTED_STRATEGY_PROFILES]
        if invalid:
            raise ValueError(f"Unsupported strategy profile(s): {', '.join(invalid)}")
        if len(unique) != 2:
            raise ValueError("Workspace batch analysis expects exactly two strategy profiles")
        if set(unique) != set(WORKSPACE_BATCH_PROFILES):
            allowed = " and ".join(WORKSPACE_BATCH_PROFILES)
            raise ValueError(f"Workspace batch analysis must run {allowed} together")
        return [profile for profile in WORKSPACE_BATCH_PROFILES if profile in unique]
