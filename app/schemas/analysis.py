from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.schemas.common import (
    Action,
    Bias,
    EmaAlignment,
    RecommendedTiming,
    StructureState,
    SupportedTimeframe,
    TriggerState,
    VolatilityState,
)


class TimeframeAnalysis(BaseModel):
    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    timeframe: SupportedTimeframe
    latest_timestamp: datetime
    close: float
    ema21: float = Field(validation_alias=AliasChoices("ema21", "ema20"))
    ema55: float = Field(validation_alias=AliasChoices("ema55", "ema50"))
    ema100: float
    ema200: float
    atr14: float
    atr_pct: float
    price_vs_ema21_pct: float = Field(validation_alias=AliasChoices("price_vs_ema21_pct", "price_vs_ema20_pct"))
    price_vs_ema55_pct: float = Field(validation_alias=AliasChoices("price_vs_ema55_pct", "price_vs_ema50_pct"))
    price_vs_ema100_pct: float
    price_vs_ema200_pct: float
    ema_alignment: EmaAlignment
    trend_bias: Bias
    trend_score: int = Field(ge=0, le=100)
    structure_state: StructureState
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    is_pullback_to_value_area: bool
    is_extended: bool
    trigger_state: TriggerState = TriggerState.NOT_APPLICABLE
    notes: list[str] = Field(default_factory=list)


class MarketRegime(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    higher_timeframe_bias: Bias
    trend_strength: int = Field(ge=0, le=100)
    volatility_state: VolatilityState
    is_trend_friendly: bool


class TimeframesAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    day_1: Optional[TimeframeAnalysis] = Field(default=None, alias="1d")
    hour_4: Optional[TimeframeAnalysis] = Field(default=None, alias="4h")
    hour_1: Optional[TimeframeAnalysis] = Field(default=None, alias="1h")
    min_15: Optional[TimeframeAnalysis] = Field(default=None, alias="15m")
    min_3: Optional[TimeframeAnalysis] = Field(default=None, alias="3m")


class ChartCandle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class ChartLinePoint(BaseModel):
    timestamp: datetime
    value: float


class TimeframeChart(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    timeframe: SupportedTimeframe
    candles: list[ChartCandle] = Field(default_factory=list)
    ema21: list[ChartLinePoint] = Field(default_factory=list)
    ema55: list[ChartLinePoint] = Field(default_factory=list)
    ema100: list[ChartLinePoint] = Field(default_factory=list)
    ema200: list[ChartLinePoint] = Field(default_factory=list)


class AnalysisCharts(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    day_1: Optional[TimeframeChart] = Field(default=None, alias="1d")
    hour_4: Optional[TimeframeChart] = Field(default=None, alias="4h")
    hour_1: Optional[TimeframeChart] = Field(default=None, alias="1h")
    min_15: Optional[TimeframeChart] = Field(default=None, alias="15m")
    min_3: Optional[TimeframeChart] = Field(default=None, alias="3m")


class EntryZone(BaseModel):
    low: float
    high: float
    basis: str


class StopLoss(BaseModel):
    price: float
    basis: str


class TakeProfitHint(BaseModel):
    tp1: float
    tp2: float
    basis: str


class Decision(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    action: Action
    bias: Bias
    confidence: int = Field(ge=0, le=100)
    recommended_timing: RecommendedTiming
    entry_zone: Optional[EntryZone] = None
    stop_loss: Optional[StopLoss] = None
    invalidation: str
    invalidation_price: Optional[float] = None
    take_profit_hint: Optional[TakeProfitHint] = None


class Reasoning(BaseModel):
    reasons_for: list[str] = Field(default_factory=list)
    reasons_against: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    state_notes: list[str] = Field(default_factory=list)
    summary: str


class ScoreContribution(BaseModel):
    label: str
    points: int
    note: str


class ScoreBreakdown(BaseModel):
    base: int = Field(ge=0, le=100)
    total: int = Field(ge=0, le=100)
    contributions: list[ScoreContribution] = Field(default_factory=list)


class SetupQuality(BaseModel):
    setup_timeframe: SupportedTimeframe = SupportedTimeframe.HOUR_1
    higher_timeframe_bias: Bias
    trend_friendly: bool
    setup_timeframe_aligned: bool = Field(
        validation_alias=AliasChoices("setup_timeframe_aligned", "mid_timeframe_aligned")
    )
    setup_timeframe_pullback_ready: bool = Field(
        validation_alias=AliasChoices("setup_timeframe_pullback_ready", "mid_timeframe_pullback_ready")
    )
    setup_timeframe_extended: bool = Field(
        validation_alias=AliasChoices("setup_timeframe_extended", "mid_timeframe_extended")
    )
    setup_distance_to_value_atr: float = Field(
        validation_alias=AliasChoices("setup_distance_to_value_atr", "one_hour_distance_to_value_atr")
    )


class TriggerMaturity(BaseModel):
    timeframe: SupportedTimeframe = SupportedTimeframe.MIN_15
    state: TriggerState
    score: int
    supporting_signals: list[str] = Field(default_factory=list)
    blocking_signals: list[str] = Field(default_factory=list)


class AnalysisDiagnostics(BaseModel):
    strategy_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    score_breakdown: ScoreBreakdown
    vetoes: list[str] = Field(default_factory=list)
    conflict_signals: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    state_notes: list[str] = Field(default_factory=list)
    setup_quality: SetupQuality
    trigger_maturity: TriggerMaturity


class AnalysisResult(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    analysis_id: str
    timestamp: datetime
    symbol: str
    exchange: str
    market_type: str
    strategy_profile: str
    timeframes: TimeframesAnalysis
    charts: Optional[AnalysisCharts] = None
    market_regime: MarketRegime
    decision: Decision
    reasoning: Reasoning
    diagnostics: AnalysisDiagnostics
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
