from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies import get_analysis_service
from app.core.database import Base, get_db
from app.main import create_app
from app.models.analysis_record import AnalysisRecord
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
    recommended_timing: str,
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
            request_payload={"timeframes": ["1d", "4h", "1h", "15m"]},
            result_payload={
                "analysis_id": analysis_id,
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "exchange": "binance",
                "market_type": "perpetual",
                "strategy_profile": strategy_profile,
                "timeframes": {
                    "1d": {
                        "timeframe": "1d",
                        "latest_timestamp": timestamp.isoformat(),
                        "close": 100.0,
                        "ema20": 99.0,
                        "ema50": 98.0,
                        "ema100": 97.0,
                        "ema200": 96.0,
                        "atr14": 2.0,
                        "atr_pct": 2.0,
                        "price_vs_ema20_pct": 1.0,
                        "price_vs_ema50_pct": 2.0,
                        "price_vs_ema100_pct": 3.0,
                        "price_vs_ema200_pct": 4.0,
                        "ema_alignment": "bullish",
                        "trend_bias": bias,
                        "trend_score": 70,
                        "structure_state": "higher_highs_higher_lows",
                        "swing_high": 110.0,
                        "swing_low": 90.0,
                        "is_pullback_to_value_area": False,
                        "is_extended": False,
                        "trigger_state": "not_applicable",
                        "notes": [],
                    },
                    "4h": {
                        "timeframe": "4h",
                        "latest_timestamp": timestamp.isoformat(),
                        "close": 100.0,
                        "ema20": 99.0,
                        "ema50": 98.0,
                        "ema100": 97.0,
                        "ema200": 96.0,
                        "atr14": 2.0,
                        "atr_pct": 2.0,
                        "price_vs_ema20_pct": 1.0,
                        "price_vs_ema50_pct": 2.0,
                        "price_vs_ema100_pct": 3.0,
                        "price_vs_ema200_pct": 4.0,
                        "ema_alignment": "bullish",
                        "trend_bias": bias,
                        "trend_score": 68,
                        "structure_state": "higher_highs_higher_lows",
                        "swing_high": 110.0,
                        "swing_low": 90.0,
                        "is_pullback_to_value_area": False,
                        "is_extended": False,
                        "trigger_state": "not_applicable",
                        "notes": [],
                    },
                    "1h": {
                        "timeframe": "1h",
                        "latest_timestamp": timestamp.isoformat(),
                        "close": 100.0,
                        "ema20": 99.0,
                        "ema50": 98.0,
                        "ema100": 97.0,
                        "ema200": 96.0,
                        "atr14": 2.0,
                        "atr_pct": 2.0,
                        "price_vs_ema20_pct": 1.0,
                        "price_vs_ema50_pct": 2.0,
                        "price_vs_ema100_pct": 3.0,
                        "price_vs_ema200_pct": 4.0,
                        "ema_alignment": "bullish",
                        "trend_bias": bias,
                        "trend_score": 65,
                        "structure_state": "higher_highs_higher_lows",
                        "swing_high": 110.0,
                        "swing_low": 90.0,
                        "is_pullback_to_value_area": False,
                        "is_extended": False,
                        "trigger_state": "not_applicable",
                        "notes": [],
                    },
                    "15m": {
                        "timeframe": "15m",
                        "latest_timestamp": timestamp.isoformat(),
                        "close": 100.0,
                        "ema20": 99.0,
                        "ema50": 98.0,
                        "ema100": 97.0,
                        "ema200": 96.0,
                        "atr14": 2.0,
                        "atr_pct": 2.0,
                        "price_vs_ema20_pct": 1.0,
                        "price_vs_ema50_pct": 2.0,
                        "price_vs_ema100_pct": 3.0,
                        "price_vs_ema200_pct": 4.0,
                        "ema_alignment": "bullish",
                        "trend_bias": bias,
                        "trend_score": 60 if action == "LONG" else 30,
                        "structure_state": "higher_highs_higher_lows" if action == "LONG" else "lower_highs_lower_lows",
                        "swing_high": 105.0,
                        "swing_low": 95.0,
                        "is_pullback_to_value_area": action == "LONG",
                        "is_extended": action == "SHORT",
                        "trigger_state": "bullish_confirmed" if action == "LONG" else "bearish_confirmed",
                        "notes": [],
                    },
                },
                "market_regime": {
                    "higher_timeframe_bias": bias,
                    "trend_strength": 71 if action == "LONG" else 54,
                    "volatility_state": "normal",
                    "is_trend_friendly": action != "WAIT",
                },
                "decision": {
                    "action": action,
                    "bias": bias,
                    "confidence": 72,
                    "recommended_timing": recommended_timing,
                    "entry_zone": None,
                    "stop_loss": None,
                    "invalidation": f"{analysis_id} invalidation",
                    "take_profit_hint": None,
                },
                "reasoning": {
                    "reasons_for": [f"{analysis_id} reason for"],
                    "reasons_against": [f"{analysis_id} reason against"] if action == "WAIT" else [],
                    "risk_notes": [f"{analysis_id} risk note"],
                    "summary": f"{analysis_id} summary",
                },
                "diagnostics": {
                    "strategy_config_snapshot": {"strategy_profile": strategy_profile},
                    "score_breakdown": {
                        "base": 50,
                        "total": 72 if action != "WAIT" else 41,
                        "contributions": [],
                    },
                    "vetoes": [],
                    "conflict_signals": [],
                    "uncertainty_notes": [],
                    "setup_quality": {
                        "higher_timeframe_bias": bias,
                        "trend_friendly": action != "WAIT",
                        "mid_timeframe_aligned": action == "LONG",
                        "mid_timeframe_pullback_ready": action == "LONG",
                        "mid_timeframe_extended": action == "SHORT",
                        "one_hour_distance_to_value_atr": 1.5,
                    },
                    "trigger_maturity": {
                        "timeframe": "15m",
                        "state": "bullish_confirmed" if action == "LONG" else "bearish_confirmed",
                        "score": 65 if action == "LONG" else 25,
                        "supporting_signals": ["regained structure"] if action == "LONG" else [],
                        "blocking_signals": ["too extended"] if action == "SHORT" else [],
                    },
                },
                "raw_metrics": {
                    "requested_timeframes": ["1d", "4h", "1h", "15m"],
                },
            },
        )
    )


@pytest.fixture()
def review_session_factory(tmp_path: Path):
    db_path = tmp_path / "review.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    try:
        yield SessionLocal
    finally:
        engine.dispose()


def _seed_review_data(db: Session) -> None:
    base_time = datetime(2026, 3, 18, 8, 0, tzinfo=timezone.utc)
    _insert_record(
        db,
        analysis_id="review-long",
        symbol="ETH/USDT:USDT",
        action="LONG",
        bias="bullish",
        strategy_profile="trend_pullback_v1",
        created_at=base_time,
        timestamp=base_time,
        recommended_timing="now",
    )
    _insert_record(
        db,
        analysis_id="review-short",
        symbol="BTC/USDT:USDT",
        action="SHORT",
        bias="bearish",
        strategy_profile="trend_pullback_v1",
        created_at=base_time + timedelta(minutes=30),
        timestamp=base_time + timedelta(minutes=30),
        recommended_timing="wait_confirmation",
    )
    db.commit()


class ReviewProxyService:
    def __init__(self) -> None:
        self.persistence = PersistenceService()
        self.persistence_service = self.persistence

    def list_analyses(self, db, **kwargs):
        return self.persistence.list_analyses(db, **kwargs)

    def get_analysis(self, db, analysis_id):
        return self.persistence.get_analysis(db, analysis_id)


def test_review_page_returns_built_in_html() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/review")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "交易复盘" in response.text
    assert "加载分析" in response.text
    assert "/analysis/" in response.text
    assert "/analyses/compare" not in response.text
    assert "左侧分析 ID" in response.text
    assert "右侧分析 ID" in response.text


def test_compare_analyses_returns_structured_diff(review_session_factory) -> None:
    seed_db = review_session_factory()
    _seed_review_data(seed_db)
    seed_db.close()

    app = create_app()
    app.dependency_overrides[get_analysis_service] = lambda: ReviewProxyService()

    def override_get_db():
        db = review_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.get("/analysis/review-long/diff/review-short")
        legacy = client.get("/analyses/compare", params={"left_id": "review-long", "right_id": "review-short"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["left"]["analysis_id"] == "review-long"
    assert payload["right"]["analysis_id"] == "review-short"
    assert payload["decision"]["changed"] is True
    assert payload["market_regime"]["changed"] is True
    assert payload["diagnostics"]["changed"] is True
    assert payload["timeframes"]
    assert any(item["timeframe"] == "15m" for item in payload["timeframes"])
    assert payload["changed_sections"][:3] == ["decision", "market_regime", "diagnostics"]
    assert payload["total_change_count"] > 0
    assert payload["summary"]

    assert legacy.status_code == 404
