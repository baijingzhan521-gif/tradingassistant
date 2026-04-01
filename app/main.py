from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_analyze import router as analyze_router
from app.api.routes_health import router as health_router
from app.api.routes_history import router as history_router
from app.api.routes_review import router as review_router
from app.api.routes_workspace_backtests import router as workspace_backtests_router
from app.api.routes_workspace_analysis import router as workspace_analysis_router
from app.api.routes_symbols import router as symbols_router
from app.api.routes_workspace import router as workspace_router
from app.core.config import get_settings
from app.core.database import init_db
from app.core.exceptions import TradingAssistantError
from app.core.logging import configure_logging


configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    logger.info("Database initialized")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Rule-based multi-timeframe trading analysis assistant. Analysis only, no order execution.",
        lifespan=lifespan,
    )

    app.include_router(workspace_router)
    app.include_router(health_router)
    app.include_router(symbols_router)
    app.include_router(analyze_router)
    app.include_router(workspace_analysis_router)
    app.include_router(workspace_backtests_router)
    app.include_router(history_router)
    app.include_router(review_router)

    @app.exception_handler(TradingAssistantError)
    def handle_trading_error(_: Request, exc: TradingAssistantError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return app


app = create_app()
