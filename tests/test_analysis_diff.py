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
from app.schemas.common import Action, Bias, EmaAlignment, RecommendedTiming, StructureState, TriggerState, VolatilityState
from app.services.persistence_service import PersistenceService


def _persist_analysis(db: Session, result, *, created_at: datetime | None = None) -> None:
    db.add(
        AnalysisRecord(
            analysis_id=result.analysis_id,
            created_at=created_at or result.timestamp,
            symbol=result.symbol,
            exchange=result.exchange,
            market_type=result.market_type,
            strategy_profile=result.strategy_profile,
            action=result.decision.action,
            bias=result.decision.bias,
            confidence=result.decision.confidence,
            summary=result.reasoning.summary,
            request_payload={"timeframes": ["1d", "4h", "1h", "15m"]},
            result_payload=result.model_dump(mode="json"),
        )
    )


@pytest.fixture()
def diff_session_factory(tmp_path: Path):
    db_path = tmp_path / "analysis_diff.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    try:
        yield SessionLocal
    finally:
        engine.dispose()


@pytest.fixture()
def diff_db(diff_session_factory):
    db = diff_session_factory()
    try:
        yield db
    finally:
        db.close()


def _build_changed_analysis(sample_analysis_result):
    timeframe_1h = sample_analysis_result.timeframes.hour_1.model_copy(
        update={
            "trend_bias": Bias.NEUTRAL,
            "ema_alignment": EmaAlignment.MIXED,
            "trend_score": 49,
            "structure_state": StructureState.MIXED,
            "is_pullback_to_value_area": False,
            "is_extended": True,
            "trigger_state": TriggerState.MIXED,
            "notes": ["Momentum cooled down", "Structure lost alignment"],
        }
    )
    timeframe_15m = sample_analysis_result.timeframes.min_15.model_copy(
        update={
            "trend_bias": Bias.NEUTRAL,
            "ema_alignment": EmaAlignment.MIXED,
            "trend_score": 36,
            "structure_state": StructureState.MIXED,
            "is_pullback_to_value_area": False,
            "is_extended": True,
            "trigger_state": TriggerState.NONE,
            "notes": ["No confirmation", "Price is chopping"],
        }
    )
    return sample_analysis_result.model_copy(
        update={
            "analysis_id": "analysis-beta",
            "timestamp": sample_analysis_result.timestamp + timedelta(minutes=30),
            "timeframes": sample_analysis_result.timeframes.model_copy(
                update={
                    "hour_1": timeframe_1h,
                    "min_15": timeframe_15m,
                }
            ),
            "market_regime": sample_analysis_result.market_regime.model_copy(
                update={
                    "higher_timeframe_bias": Bias.NEUTRAL,
                    "trend_strength": 41,
                    "volatility_state": VolatilityState.HIGH,
                    "is_trend_friendly": False,
                }
            ),
            "decision": sample_analysis_result.decision.model_copy(
                update={
                    "action": Action.WAIT,
                    "bias": Bias.NEUTRAL,
                    "confidence": 18,
                    "recommended_timing": RecommendedTiming.SKIP,
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": "No valid setup after higher-timeframe alignment broke",
                }
            ),
            "reasoning": sample_analysis_result.reasoning.model_copy(
                update={
                    "reasons_for": ["Price structure weakened"],
                    "reasons_against": ["Higher timeframe lost alignment", "15m confirmation missing"],
                    "risk_notes": ["Volatility increased", "No actionable trigger"],
                    "summary": "The setup degrades into a wait state after losing multi-timeframe alignment.",
                }
            ),
            "diagnostics": sample_analysis_result.diagnostics.model_copy(
                update={
                    "strategy_config_snapshot": {
                        "atr_period": 14,
                        "extended_threshold_atr": 2.0,
                        "pullback_threshold_atr": 0.4,
                    },
                    "score_breakdown": sample_analysis_result.diagnostics.score_breakdown.model_copy(
                        update={
                            "base": 50,
                            "total": 18,
                            "contributions": [],
                        }
                    ),
                    "vetoes": ["Higher timeframe lost alignment"],
                    "conflict_signals": ["15m trigger is not confirmed"],
                    "uncertainty_notes": ["Trend extension is too large", "Confirmation is incomplete"],
                    "setup_quality": sample_analysis_result.diagnostics.setup_quality.model_copy(
                        update={
                            "higher_timeframe_bias": Bias.NEUTRAL,
                            "trend_friendly": False,
                            "setup_timeframe_aligned": False,
                            "setup_timeframe_pullback_ready": False,
                            "setup_timeframe_extended": True,
                            "setup_distance_to_value_atr": 2.3,
                        }
                    ),
                    "trigger_maturity": sample_analysis_result.diagnostics.trigger_maturity.model_copy(
                        update={
                            "state": TriggerState.NONE,
                            "score": 2,
                            "supporting_signals": [],
                            "blocking_signals": ["No 15m confirmation"],
                        }
                    ),
                }
            ),
            "raw_metrics": {
                "scorecard": {"base": 50, "total": 18, "contributions": []},
                "volatility": {"state": "high"},
            },
        }
    )


def _seed_diff_history(db: Session, sample_analysis_result) -> None:
    _persist_analysis(db, sample_analysis_result, created_at=sample_analysis_result.timestamp)
    _persist_analysis(db, _build_changed_analysis(sample_analysis_result), created_at=sample_analysis_result.timestamp + timedelta(minutes=30))
    db.commit()


def test_compare_analyses_returns_structured_diff(diff_db: Session, sample_analysis_result) -> None:
    _seed_diff_history(diff_db, sample_analysis_result)
    service = PersistenceService()

    diff = service.compare_analyses(diff_db, "test-analysis-id", "analysis-beta")

    assert diff.left.analysis_id == "test-analysis-id"
    assert diff.right.analysis_id == "analysis-beta"
    assert diff.same_symbol is True
    assert diff.decision.changed is True
    assert diff.market_regime.changed is True
    assert diff.diagnostics.changed is True
    assert diff.changed_sections[:3] == ["decision", "market_regime", "diagnostics"]
    assert any(item.timeframe == "1h" for item in diff.timeframes)
    assert any(item.timeframe == "15m" for item in diff.timeframes)
    assert any(change.field == "decision.action" for change in diff.decision.changed_fields)
    assert any(change.field == "market_regime.higher_timeframe_bias" for change in diff.market_regime.changed_fields)
    assert any(change.field == "diagnostics.score_breakdown.total" for change in diff.diagnostics.changed_fields)

    one_hour_diff = next(item for item in diff.timeframes if item.timeframe == "1h")
    assert any(change.field.endswith("trend_score") for change in one_hour_diff.changed_fields)
    assert one_hour_diff.signal_shift is not None
    assert diff.total_change_count > 0
    assert "decision.action" in diff.summary or "market_regime" in diff.summary


def test_compare_analyses_route_and_not_found(diff_session_factory, sample_analysis_result) -> None:
    seed_db = diff_session_factory()
    _seed_diff_history(seed_db, sample_analysis_result)
    seed_db.close()

    app = create_app()
    app.dependency_overrides[get_analysis_service] = lambda: HistoryProxyService()

    def override_get_db():
        db = diff_session_factory()
        try:
            yield db
        finally:
            db.close()

    from app.core.database import get_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.get("/analysis/test-analysis-id/diff/analysis-beta")
        missing = client.get("/analysis/test-analysis-id/diff/missing-analysis")

    assert response.status_code == 200
    payload = response.json()
    assert payload["left"]["analysis_id"] == "test-analysis-id"
    assert payload["right"]["analysis_id"] == "analysis-beta"
    assert payload["decision"]["changed"] is True
    assert payload["market_regime"]["changed"] is True
    assert payload["diagnostics"]["changed"] is True
    assert any(item["timeframe"] == "15m" for item in payload["timeframes"])
    assert payload["changed_sections"][:3] == ["decision", "market_regime", "diagnostics"]

    assert missing.status_code == 404
    assert "Analysis not found" in missing.json()["detail"]


class HistoryProxyService:
    def __init__(self) -> None:
        self.persistence = PersistenceService()
        self.persistence_service = self.persistence

    def list_analyses(self, db, **kwargs):
        return self.persistence.list_analyses(db, **kwargs)

    def get_analysis(self, db, analysis_id):
        return self.persistence.get_analysis(db, analysis_id)
