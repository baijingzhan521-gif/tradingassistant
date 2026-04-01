from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base
from app.models.derivatives_liquidation import DerivativesLiquidationEvent, DerivativesLiquidationHour
from app.services.derivatives_liquidation_service import DerivativesLiquidationService


def make_session(tmp_path) -> Session:
    db_path = tmp_path / "liquidation.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return SessionLocal()


def test_normalize_bybit_payload_builds_event_rows() -> None:
    service = DerivativesLiquidationService()
    rows = service.normalize_bybit_payload(
        {
            "topic": "allLiquidation.BTCUSDT",
            "type": "snapshot",
            "ts": 1710000005000,
            "data": [
                {"T": 1710000000000, "s": "BTCUSDT", "S": "Sell", "v": "0.25", "p": "51000"},
            ],
        }
    )

    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTCUSDT"
    assert rows[0]["side"] == "Sell"
    assert rows[0]["notional_usd"] == 12750.0
    assert rows[0]["message_timestamp"] == datetime(2024, 3, 9, 16, 0, 5, tzinfo=timezone.utc)


def test_persist_events_and_rebuild_hourly_aggregates(tmp_path) -> None:
    service = DerivativesLiquidationService()
    db = make_session(tmp_path)
    try:
        inserted = service.persist_events(
            db,
            [
                {
                    "venue": "bybit",
                    "symbol": "BTCUSDT",
                    "source_topic": "allLiquidation.BTCUSDT",
                    "message_type": "snapshot",
                    "message_timestamp": datetime(2024, 3, 9, 16, 0, 5, tzinfo=timezone.utc),
                    "event_timestamp": datetime(2024, 3, 9, 16, 0, 0, tzinfo=timezone.utc),
                    "side": "Buy",
                    "size": 0.5,
                    "bankruptcy_price": 50000.0,
                    "notional_usd": 25000.0,
                    "created_at": datetime.now(timezone.utc),
                },
                {
                    "venue": "bybit",
                    "symbol": "BTCUSDT",
                    "source_topic": "allLiquidation.BTCUSDT",
                    "message_type": "snapshot",
                    "message_timestamp": datetime(2024, 3, 9, 16, 10, 5, tzinfo=timezone.utc),
                    "event_timestamp": datetime(2024, 3, 9, 16, 10, 0, tzinfo=timezone.utc),
                    "side": "Sell",
                    "size": 0.25,
                    "bankruptcy_price": 52000.0,
                    "notional_usd": 13000.0,
                    "created_at": datetime.now(timezone.utc),
                },
            ],
        )
        rebuilt = service.rebuild_hourly_aggregates(
            db,
            venue="bybit",
            symbol="BTCUSDT",
            interval="1h",
            start=datetime(2024, 3, 9, 16, 0, tzinfo=timezone.utc),
            end=datetime(2024, 3, 9, 17, 0, tzinfo=timezone.utc),
        )

        event_count = db.execute(select(DerivativesLiquidationEvent)).all()
        hourly = db.execute(select(DerivativesLiquidationHour)).scalar_one()
    finally:
        db.close()

    assert inserted == 2
    assert rebuilt == 1
    assert len(event_count) == 2
    assert hourly.event_count == 2
    assert hourly.buy_count == 1
    assert hourly.sell_count == 1
    assert hourly.notional_usd == 38000.0
    assert hourly.sell_minus_buy_notional_usd == -12000.0
