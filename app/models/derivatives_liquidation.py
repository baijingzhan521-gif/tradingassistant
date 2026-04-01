from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DerivativesLiquidationEvent(Base):
    __tablename__ = "derivatives_liquidation_events"
    __table_args__ = (
        UniqueConstraint(
            "venue",
            "symbol",
            "event_timestamp",
            "side",
            "size",
            "bankruptcy_price",
            "message_timestamp",
            name="uq_derivatives_liquidation_event",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    source_topic: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    side: Mapped[str] = mapped_column(String(8), index=True)
    size: Mapped[float] = mapped_column(Float)
    bankruptcy_price: Mapped[float] = mapped_column(Float)
    notional_usd: Mapped[float] = mapped_column(Float, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class DerivativesLiquidationHour(Base):
    __tablename__ = "derivatives_liquidation_hours"
    __table_args__ = (
        UniqueConstraint("venue", "symbol", "interval", "timestamp", name="uq_derivatives_liquidation_hour"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    buy_count: Mapped[int] = mapped_column(Integer, default=0)
    sell_count: Mapped[int] = mapped_column(Integer, default=0)
    notional_usd: Mapped[float] = mapped_column(Float, default=0.0)
    buy_notional_usd: Mapped[float] = mapped_column(Float, default=0.0)
    sell_notional_usd: Mapped[float] = mapped_column(Float, default=0.0)
    sell_minus_buy_notional_usd: Mapped[float] = mapped_column(Float, default=0.0)
    max_event_notional_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        index=True,
    )
