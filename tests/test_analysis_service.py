from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from app.schemas.request import AnalyzeRequest
from app.services.analysis_service import AnalysisService


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=220, freq="min", tz="UTC"),
            "open": [100.0] * 220,
            "high": [101.0] * 220,
            "low": [99.0] * 220,
            "close": [100.0] * 220,
            "volume": [1000.0] * 220,
        }
    )


class FakeOhlcvService:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def fetch_multi_timeframe(self, *, exchange: str, market_type: str, symbol: str, timeframes: list[str], limit: int):
        self.calls.append(list(timeframes))
        return {timeframe: _frame() for timeframe in timeframes}


class FakeStrategyService:
    def __init__(self) -> None:
        self.received: dict[str, pd.DataFrame] | None = None

    def run(self, request: AnalyzeRequest, ohlcv_by_timeframe: dict[str, pd.DataFrame]):
        self.received = ohlcv_by_timeframe
        return SimpleNamespace()


class FakePersistenceService:
    def save_analysis(self, db, request, analysis):
        return analysis


def test_analysis_service_fetches_all_required_fetch_timeframes_for_trend_pullback_v1() -> None:
    ohlcv_service = FakeOhlcvService()
    strategy_service = FakeStrategyService()
    persistence_service = FakePersistenceService()
    service = AnalysisService(ohlcv_service, strategy_service, persistence_service)

    request = AnalyzeRequest(
        symbol="ETH/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h", "15m"],
        strategy_profile="trend_pullback_v1",
        lookback=300,
    )

    service.analyze(request, db=None)

    assert ohlcv_service.calls == [["1d", "4h", "1h", "15m", "3m"]]
    assert strategy_service.received is not None
    assert set(strategy_service.received) == {"1d", "4h", "1h", "15m", "3m"}
