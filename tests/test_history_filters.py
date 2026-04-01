from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies import get_analysis_service
from app.core.database import Base
from app.main import create_app
from app.models.analysis_record import AnalysisRecord
from app.schemas.common import Action, Bias
from app.services.persistence_service import PersistenceService


def _insert_record(
    db: Session,
    *,
    analysis_id: str,
    symbol: str,
    action: str,
    bias: str,
    strategy_profile: str,
    created_at: datetime,
    timestamp: datetime,
    timeframes: list[str],
    recommended_timing: str = "now",
) -> None:
    db.add(
        AnalysisRecord(
            analysis_id=analysis_id,
            created_at=created_at,
            symbol=symbol,
            exchange="binance",
            market_type="perpetual",
            strategy_profile=strategy_profile,
            action=action,
            bias=bias,
            confidence=72,
            summary=f"{analysis_id} summary",
            request_payload={"timeframes": timeframes},
            result_payload={
                "analysis_id": analysis_id,
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "exchange": "binance",
                "market_type": "perpetual",
                "strategy_profile": strategy_profile,
                "decision": {
                    "action": action,
                    "bias": bias,
                    "confidence": 72,
                    "recommended_timing": recommended_timing,
                },
                "market_regime": {
                    "higher_timeframe_bias": bias,
                    "trend_strength": 75,
                    "volatility_state": "normal",
                    "is_trend_friendly": True,
                },
                "reasoning": {"summary": f"{analysis_id} summary"},
            },
        )
    )


@pytest.fixture()
def history_session_factory(tmp_path: Path):
    db_path = tmp_path / "history.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    try:
        yield SessionLocal
    finally:
        engine.dispose()


@pytest.fixture()
def history_db(history_session_factory):
    db = history_session_factory()
    try:
        yield db
    finally:
        db.close()


def _seed_history(db: Session) -> None:
    base_time = datetime(2026, 3, 18, 8, 0, tzinfo=timezone.utc)
    _insert_record(
        db,
        analysis_id="a1",
        symbol="ETH/USDT:USDT",
        action="LONG",
        bias="bullish",
        strategy_profile="trend_pullback_v1",
        created_at=base_time,
        timestamp=base_time,
        timeframes=["1d", "4h", "1h", "15m"],
        recommended_timing="now",
    )
    _insert_record(
        db,
        analysis_id="a2",
        symbol="ETH/USDT:USDT",
        action="WAIT",
        bias="neutral",
        strategy_profile="trend_pullback_v1",
        created_at=base_time + timedelta(minutes=30),
        timestamp=base_time + timedelta(minutes=30),
        timeframes=["1d", "4h", "1h", "15m"],
        recommended_timing="skip",
    )
    _insert_record(
        db,
        analysis_id="b1",
        symbol="BTC/USDT:USDT",
        action="SHORT",
        bias="bearish",
        strategy_profile="trend_pullback_v1",
        created_at=base_time + timedelta(hours=1),
        timestamp=base_time + timedelta(hours=1),
        timeframes=["1d", "4h", "1h", "15m"],
        recommended_timing="now",
    )
    db.commit()


class HistoryProxyService:
    def __init__(self) -> None:
        self.persistence = PersistenceService()
        self.persistence_service = self.persistence

    def list_analyses(self, db, **kwargs):
        return self.persistence.list_analyses(db, **kwargs)

    def get_analysis(self, db, analysis_id):
        return self.persistence.get_analysis(db, analysis_id)


def test_list_analyses_filters_and_pagination(history_db: Session) -> None:
    _seed_history(history_db)
    service = PersistenceService()

    response = service.list_analyses(
        history_db,
        limit=2,
        offset=0,
        symbol="ETH/USDT:USDT",
        action=Action.LONG,
        bias=Bias.BULLISH,
        strategy_profile="trend_pullback_v1",
        from_time=datetime(2026, 3, 18, 7, 0, tzinfo=timezone.utc),
        to_time=datetime(2026, 3, 18, 9, 0, tzinfo=timezone.utc),
    )

    assert response.total == 1
    assert response.pagination.returned == 1
    assert response.pagination.has_more is False
    assert response.items[0].analysis_id == "a1"
    assert response.items[0].requested_timeframes == ["1d", "4h", "1h", "15m"]
    assert response.items[0].recommended_timing == "now"
    assert response.items[0].higher_timeframe_bias == "bullish"


def test_analyses_route_applies_filters(history_session_factory) -> None:
    seed_db = history_session_factory()
    _seed_history(seed_db)
    seed_db.close()

    app = create_app()
    app.dependency_overrides[get_analysis_service] = lambda: HistoryProxyService()

    def override_get_db():
        db = history_session_factory()
        try:
            yield db
        finally:
            db.close()

    from app.core.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.get(
            "/analyses",
            params={
                "limit": 10,
                "offset": 0,
                "symbol": "BTC/USDT:USDT",
                "action": "SHORT",
                "bias": "bearish",
                "strategy_profile": "trend_pullback_v1",
                "from_time": "2026-03-18T08:30:00Z",
                "to_time": "2026-03-18T10:00:00Z",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["total"] == 1
    assert payload["pagination"]["returned"] == 1
    assert payload["items"][0]["analysis_id"] == "b1"
    assert payload["items"][0]["symbol"] == "BTC/USDT:USDT"
    assert payload["items"][0]["action"] == "SHORT"
