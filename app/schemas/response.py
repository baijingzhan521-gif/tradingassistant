from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.analysis import AnalysisResult
from app.schemas.common import Action, Bias, RecommendedTiming, VolatilityState


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


class SymbolsResponse(BaseModel):
    exchange: str
    market_type: str
    count: int
    symbols: list[str]


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    analysis_id: str
    timestamp: datetime
    recorded_at: datetime
    symbol: str
    exchange: str
    market_type: str
    strategy_profile: str
    requested_timeframes: list[str] = Field(default_factory=list)
    action: Action
    bias: Bias
    confidence: int
    recommended_timing: RecommendedTiming
    higher_timeframe_bias: Bias
    trend_strength: int
    volatility_state: VolatilityState
    is_trend_friendly: bool
    summary: str


class AnalysisListPagination(BaseModel):
    limit: int
    offset: int
    total: int
    returned: int
    has_more: bool
    next_offset: Optional[int] = None
    previous_offset: Optional[int] = None


class AnalysisListResponse(BaseModel):
    items: list[AnalysisSummary] = Field(default_factory=list)
    total: int
    pagination: AnalysisListPagination = Field(
        default_factory=lambda: AnalysisListPagination(
            limit=0,
            offset=0,
            total=0,
            returned=0,
            has_more=False,
        )
    )


class FieldChange(BaseModel):
    field: str
    before: Any = None
    after: Any = None
    changed: bool = True
    delta: Optional[float] = None
    pct_change: Optional[float] = None
    added: list[Any] = Field(default_factory=list)
    removed: list[Any] = Field(default_factory=list)


class SectionDiff(BaseModel):
    changed: bool
    change_count: int
    changed_fields: list[FieldChange] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)


class TimeframeDiff(SectionDiff):
    timeframe: str
    signal_shift: Optional[str] = None


class AnalysisDiffResponse(BaseModel):
    left: AnalysisSummary
    right: AnalysisSummary
    same_symbol: bool
    same_exchange: bool
    same_market_type: bool
    same_strategy_profile: bool
    compared_at: datetime
    decision: SectionDiff
    market_regime: SectionDiff
    diagnostics: SectionDiff
    timeframes: list[TimeframeDiff] = Field(default_factory=list)
    changed_sections: list[str] = Field(default_factory=list)
    total_change_count: int
    summary: str


class WorkspaceBatchAnalysisItem(BaseModel):
    strategy_profile: str
    analysis: AnalysisResult


class WorkspaceBatchAnalysisResponse(BaseModel):
    batch_id: str
    timestamp: datetime
    symbol: str
    exchange: str
    market_type: str
    strategy_profiles: list[str] = Field(default_factory=list)
    analyses: list[WorkspaceBatchAnalysisItem] = Field(default_factory=list)


class BacktestTradeSnapshot(BaseModel):
    sequence: int
    symbol: str
    strategy_profile: str
    side: str
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    bars_held: int
    exit_reason: str
    confidence: int
    tp1_hit: bool
    tp2_hit: bool
    pnl_pct: float
    pnl_r: float
    cumulative_r_after_trade: float


class WorkspaceBacktestTradeBookResponse(BaseModel):
    dataset: str
    dataset_generated_at: str
    symbol: str
    strategy_profile: str
    swing_detection_mode: str
    exit_profile_label: str
    total_trades: int
    profit_factor: float
    expectancy_r: float
    cumulative_r: float
    max_drawdown_r: float
    long_trades: int
    long_r: float
    short_trades: int
    short_r: float
    ranking_source_csv: str
    trades_source_csv: str
    trades: list[BacktestTradeSnapshot] = Field(default_factory=list)
