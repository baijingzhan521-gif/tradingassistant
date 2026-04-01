from __future__ import annotations

from functools import lru_cache

from app.data.exchange_client import ExchangeClientFactory, get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.analysis_service import AnalysisService
from app.services.backtest_artifact_service import BacktestArtifactService
from app.services.persistence_service import PersistenceService
from app.services.strategy_service import StrategyService


@lru_cache(maxsize=1)
def get_ohlcv_service() -> OhlcvService:
    return OhlcvService(get_exchange_client_factory())


@lru_cache(maxsize=1)
def get_strategy_service() -> StrategyService:
    return StrategyService()


@lru_cache(maxsize=1)
def get_persistence_service() -> PersistenceService:
    return PersistenceService()


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    return AnalysisService(
        ohlcv_service=get_ohlcv_service(),
        strategy_service=get_strategy_service(),
        persistence_service=get_persistence_service(),
    )


@lru_cache(maxsize=1)
def get_backtest_artifact_service() -> BacktestArtifactService:
    return BacktestArtifactService()


def get_symbol_exchange_factory() -> ExchangeClientFactory:
    return get_exchange_client_factory()
