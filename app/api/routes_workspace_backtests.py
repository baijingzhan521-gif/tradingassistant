from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_backtest_artifact_service
from app.schemas.response import WorkspaceBacktestTradeBookResponse
from app.services.backtest_artifact_service import BacktestArtifactService


router = APIRouter(tags=["workspace"])


@router.get("/workspace/backtest-trades/btc-best", response_model=WorkspaceBacktestTradeBookResponse)
def get_best_btc_backtest_trade_book(
    artifact_service: BacktestArtifactService = Depends(get_backtest_artifact_service),
) -> WorkspaceBacktestTradeBookResponse:
    return artifact_service.get_best_btc_trade_book()
