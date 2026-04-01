from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_analysis_service
from app.core.database import get_db
from app.schemas.analysis import AnalysisResult
from app.schemas.request import AnalyzeRequest
from app.services.analysis_service import AnalysisService


router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResult, status_code=status.HTTP_201_CREATED)
def analyze(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResult:
    return analysis_service.analyze(request, db)
