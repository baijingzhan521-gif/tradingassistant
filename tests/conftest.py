from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.analysis import (
    AnalysisDiagnostics,
    AnalysisResult,
    Decision,
    EntryZone,
    MarketRegime,
    Reasoning,
    ScoreBreakdown,
    ScoreContribution,
    SetupQuality,
    StopLoss,
    TakeProfitHint,
    TimeframeAnalysis,
    TimeframesAnalysis,
    TriggerMaturity,
)
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


def make_ohlcv_frame(
    *,
    periods: int,
    start_price: float,
    up_step: float,
    freq: str,
    pullback_step: float = 0.0,
    pullback_len: int = 0,
    recovery_step: float = 0.0,
    recovery_len: int = 0,
) -> pd.DataFrame:
    close_values: list[float] = []
    price = start_price
    pullback_start = periods - pullback_len - recovery_len
    recovery_start = periods - recovery_len

    for idx in range(periods):
        if pullback_len and pullback_start <= idx < recovery_start:
            price += pullback_step
        elif recovery_len and idx >= recovery_start:
            price += recovery_step
        else:
            price += up_step
        close_values.append(price)

    timestamps = pd.date_range("2025-01-01", periods=periods, freq=freq, tz="UTC")
    opens = [close_values[0] - 0.5] + close_values[:-1]
    highs = [max(open_price, close_price) + 0.8 for open_price, close_price in zip(opens, close_values)]
    lows = [min(open_price, close_price) - 0.8 for open_price, close_price in zip(opens, close_values)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close_values,
            "volume": [1000 + idx for idx in range(periods)],
        }
    )


@pytest.fixture
def app_instance():
    return create_app()


@pytest.fixture
def client(app_instance):
    with TestClient(app_instance) as test_client:
        yield test_client


@pytest.fixture
def sample_analysis_result() -> AnalysisResult:
    timeframe_1h = TimeframeAnalysis(
        timeframe=SupportedTimeframe.HOUR_1,
        latest_timestamp=datetime.now(timezone.utc),
        close=3500.0,
        ema21=3480.0,
        ema55=3450.0,
        ema100=3400.0,
        ema200=3300.0,
        atr14=45.0,
        atr_pct=1.28,
        price_vs_ema21_pct=0.57,
        price_vs_ema55_pct=1.44,
        price_vs_ema100_pct=2.94,
        price_vs_ema200_pct=6.06,
        ema_alignment=EmaAlignment.BULLISH,
        trend_bias=Bias.BULLISH,
        trend_score=78,
        structure_state=StructureState.BULLISH,
        swing_high=3550.0,
        swing_low=3410.0,
        is_pullback_to_value_area=True,
        is_extended=False,
        trigger_state=TriggerState.BULLISH_CONFIRMED,
        notes=["Synthetic bullish test context"],
    )
    timeframe_1d = timeframe_1h.model_copy(update={"timeframe": SupportedTimeframe.DAY_1})
    timeframe_4h = timeframe_1h.model_copy(update={"timeframe": SupportedTimeframe.HOUR_4})
    timeframe_15m = timeframe_1h.model_copy(update={"timeframe": SupportedTimeframe.MIN_15})

    return AnalysisResult(
        analysis_id="test-analysis-id",
        timestamp=datetime.now(timezone.utc),
        symbol="ETH/USDT:USDT",
        exchange="binance",
        market_type="perpetual",
        strategy_profile="trend_pullback_v1",
        timeframes=TimeframesAnalysis(
            day_1=timeframe_1d,
            hour_4=timeframe_4h,
            hour_1=timeframe_1h,
            min_15=timeframe_15m,
        ),
        market_regime=MarketRegime(
            higher_timeframe_bias=Bias.BULLISH,
            trend_strength=78,
            volatility_state=VolatilityState.NORMAL,
            is_trend_friendly=True,
        ),
        decision=Decision(
            action=Action.LONG,
            bias=Bias.BULLISH,
            confidence=74,
            recommended_timing=RecommendedTiming.NOW,
            entry_zone=EntryZone(
                low=3450.0,
                high=3485.0,
                basis="1H EMA21/EMA55 pullback zone",
            ),
            stop_loss=StopLoss(price=3410.0, basis="Below recent 1H swing low"),
            invalidation="1H closes below the recent swing low",
            take_profit_hint=TakeProfitHint(
                tp1=3540.0,
                tp2=3600.0,
                basis="1R/2R and prior swing high",
            ),
        ),
        reasoning=Reasoning(
            reasons_for=["4H price above EMA200", "1H pullback to EMA55", "15m reclaimed short-term structure"],
            reasons_against=["signal still early"],
            risk_notes=["volatility normal"],
            summary="Synthetic bullish scenario for API tests.",
        ),
        diagnostics=AnalysisDiagnostics(
            strategy_config_snapshot={"atr_period": 14},
            score_breakdown=ScoreBreakdown(
                base=50,
                total=74,
                contributions=[ScoreContribution(label="1d_4h_bias", points=15, note="bullish alignment")],
            ),
            vetoes=[],
            conflict_signals=[],
            uncertainty_notes=["signal still early"],
            setup_quality=SetupQuality(
                setup_timeframe=SupportedTimeframe.HOUR_1,
                higher_timeframe_bias=Bias.BULLISH,
                trend_friendly=True,
                setup_timeframe_aligned=True,
                setup_timeframe_pullback_ready=True,
                setup_timeframe_extended=False,
                setup_distance_to_value_atr=0.2,
            ),
            trigger_maturity=TriggerMaturity(
                state=TriggerState.BULLISH_CONFIRMED,
                score=10,
                supporting_signals=["15m reclaimed EMA21"],
                blocking_signals=[],
            ),
        ),
        raw_metrics={"scorecard": {"base": 50, "total": 74, "contributions": []}},
    )
