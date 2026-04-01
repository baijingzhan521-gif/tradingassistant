from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DerivativesStateBar(Base):
    __tablename__ = "derivatives_state_bars"
    __table_args__ = (
        UniqueConstraint("venue", "symbol", "interval", "timestamp", name="uq_derivatives_state_bar"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    mark_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    index_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    premium_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    funding_rate_event: Mapped[float | None] = mapped_column(Float, nullable=True)
    funding_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest_notional_proxy_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest_change_1h_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    basis_proxy_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    mark_index_spread: Mapped[float | None] = mapped_column(Float, nullable=True)
    mark_index_spread_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
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
