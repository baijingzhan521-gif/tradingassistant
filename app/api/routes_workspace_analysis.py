from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_analysis_service
from app.core.database import get_db
from app.schemas.request import WorkspaceBatchAnalyzeRequest
from app.schemas.response import WorkspaceBatchAnalysisResponse
from app.services.analysis_service import AnalysisService


router = APIRouter(tags=["workspace"])


@router.post("/workspace/analyze/batch", response_model=WorkspaceBatchAnalysisResponse, status_code=status.HTTP_201_CREATED)
def analyze_workspace_batch(
    request: WorkspaceBatchAnalyzeRequest,
    db: Session = Depends(get_db),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> WorkspaceBatchAnalysisResponse:
    return analysis_service.analyze_batch(request, db)
