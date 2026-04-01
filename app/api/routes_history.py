from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_analysis_service
from app.core.database import get_db
from app.schemas.common import Action, Bias
from app.schemas.analysis import AnalysisResult
from app.schemas.response import AnalysisDiffResponse, AnalysisListResponse
from app.services.analysis_service import AnalysisService


router = APIRouter(tags=["history"])


@router.get("/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResult:
    return analysis_service.get_analysis(db, analysis_id)


@router.get("/analysis/{analysis_id}/diff/{comparison_analysis_id}", response_model=AnalysisDiffResponse)
def compare_analyses(
    analysis_id: str,
    comparison_analysis_id: str,
    db: Session = Depends(get_db),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisDiffResponse:
    return analysis_service.persistence_service.compare_analyses(db, analysis_id, comparison_analysis_id)


@router.get("/analyses", response_model=AnalysisListResponse)
def list_analyses(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    symbol: str | None = Query(default=None),
    action: Action | None = Query(default=None),
    bias: Bias | None = Query(default=None),
    strategy_profile: str | None = Query(default=None),
    from_time: datetime | None = Query(default=None),
    to_time: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisListResponse:
    return analysis_service.persistence_service.list_analyses(
        db,
        limit=limit,
        offset=offset,
        symbol=symbol,
        action=action,
        bias=bias,
        strategy_profile=strategy_profile,
        from_time=from_time,
        to_time=to_time,
    )
