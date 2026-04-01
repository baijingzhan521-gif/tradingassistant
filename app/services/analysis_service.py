from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from app.data.ohlcv_service import OhlcvService
from app.schemas.analysis import AnalysisResult
from app.schemas.request import AnalyzeRequest, WorkspaceBatchAnalyzeRequest
from app.schemas.response import AnalysisListResponse, WorkspaceBatchAnalysisItem, WorkspaceBatchAnalysisResponse
from app.services.persistence_service import PersistenceService
from app.services.strategy_service import StrategyService
from app.utils.timeframes import get_strategy_fetch_timeframes


logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(
        self,
        ohlcv_service: OhlcvService,
        strategy_service: StrategyService,
        persistence_service: PersistenceService,
    ) -> None:
        self.ohlcv_service = ohlcv_service
        self.strategy_service = strategy_service
        self.persistence_service = persistence_service

    def analyze(self, request: AnalyzeRequest, db: Session) -> AnalysisResult:
        logger.info(
            "Analyzing symbol=%s exchange=%s market_type=%s strategy=%s lookback=%s",
            request.symbol,
            request.exchange,
            request.market_type,
            request.strategy_profile,
            request.lookback,
        )
        fetch_timeframes = list(get_strategy_fetch_timeframes(request.strategy_profile))
        ohlcv_by_timeframe = self.ohlcv_service.fetch_multi_timeframe(
            exchange=request.exchange,
            market_type=request.market_type,
            symbol=request.symbol,
            timeframes=fetch_timeframes,
            limit=request.lookback,
        )
        analysis = self.strategy_service.run(request, ohlcv_by_timeframe)
        return self.persistence_service.save_analysis(db, request, analysis)

    def analyze_batch(self, request: WorkspaceBatchAnalyzeRequest, db: Session) -> WorkspaceBatchAnalysisResponse:
        logger.info(
            "Batch analyzing symbol=%s exchange=%s market_type=%s profiles=%s lookback=%s",
            request.symbol,
            request.exchange,
            request.market_type,
            ",".join(request.strategy_profiles),
            request.lookback,
        )
        fetch_timeframes = sorted(
            {
                timeframe
                for profile in request.strategy_profiles
                for timeframe in get_strategy_fetch_timeframes(profile)
            },
            key=lambda item: {"1d": 1440, "4h": 240, "1h": 60, "15m": 15, "3m": 3}[item],
            reverse=True,
        )
        ohlcv_by_timeframe = self.ohlcv_service.fetch_multi_timeframe(
            exchange=request.exchange,
            market_type=request.market_type,
            symbol=request.symbol,
            timeframes=fetch_timeframes,
            limit=request.lookback,
        )

        analyses: list[WorkspaceBatchAnalysisItem] = []
        for profile in request.strategy_profiles:
            profile_request = AnalyzeRequest(
                symbol=request.symbol,
                market_type=request.market_type,
                exchange=request.exchange,
                strategy_profile=profile,
                timeframes=list(get_strategy_fetch_timeframes(profile)),
                lookback=request.lookback,
            )
            analysis = self.strategy_service.run(profile_request, ohlcv_by_timeframe)
            persisted = self.persistence_service.save_analysis(db, profile_request, analysis)
            analyses.append(WorkspaceBatchAnalysisItem(strategy_profile=profile, analysis=persisted))

        return WorkspaceBatchAnalysisResponse(
            batch_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            symbol=request.symbol,
            exchange=request.exchange,
            market_type=request.market_type,
            strategy_profiles=list(request.strategy_profiles),
            analyses=analyses,
        )

    def get_analysis(self, db: Session, analysis_id: str) -> AnalysisResult:
        return self.persistence_service.get_analysis(db, analysis_id)

    def list_analyses(self, db: Session, limit: int = 50, offset: int = 0) -> AnalysisListResponse:
        return self.persistence_service.list_analyses(db, limit=limit, offset=offset)
