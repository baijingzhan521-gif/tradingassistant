from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models.derivatives_liquidation import DerivativesLiquidationEvent, DerivativesLiquidationHour


logger = logging.getLogger(__name__)

HOUR = timedelta(hours=1)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _from_millis(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)


def _floor_to_hour(value: datetime) -> datetime:
    value = _to_utc(value)
    return value.replace(minute=0, second=0, microsecond=0)


class DerivativesLiquidationService:
    def normalize_bybit_payload(self, payload: dict[str, Any], *, venue: str = "bybit") -> list[dict[str, Any]]:
        topic = str(payload.get("topic") or "")
        data = payload.get("data") or []
        if not topic.startswith("allLiquidation.") or not isinstance(data, list):
            return []

        message_timestamp = _from_millis(payload["ts"]) if payload.get("ts") is not None else None
        message_type = str(payload.get("type") or "")
        created_at = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        for item in data:
            try:
                symbol = str(item["s"])
                side = str(item["S"])
                event_timestamp = _from_millis(item["T"])
                size = float(item["v"])
                bankruptcy_price = float(item["p"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed liquidation row payload=%s error=%s", item, exc)
                continue

            rows.append(
                {
                    "venue": venue,
                    "symbol": symbol,
                    "source_topic": topic,
                    "message_type": message_type or None,
                    "message_timestamp": message_timestamp,
                    "event_timestamp": event_timestamp,
                    "side": side,
                    "size": size,
                    "bankruptcy_price": bankruptcy_price,
                    "notional_usd": size * bankruptcy_price,
                    "created_at": created_at,
                }
            )

        return rows

    def persist_events(self, db: Session, records: list[dict[str, Any]], *, chunk_size: int = 250) -> int:
        if not records:
            return 0

        inserted = 0
        for offset in range(0, len(records), chunk_size):
            chunk = records[offset : offset + chunk_size]
            stmt = sqlite_insert(DerivativesLiquidationEvent).values(chunk)
            result = db.execute(
                stmt.on_conflict_do_nothing(
                    index_elements=[
                        "venue",
                        "symbol",
                        "event_timestamp",
                        "side",
                        "size",
                        "bankruptcy_price",
                        "message_timestamp",
                    ]
                )
            )
            inserted += int(result.rowcount or 0)
        db.commit()
        return inserted

    def build_hourly_aggregate_frame(self, records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(records).copy()
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "event_count",
                    "buy_count",
                    "sell_count",
                    "notional_usd",
                    "buy_notional_usd",
                    "sell_notional_usd",
                    "sell_minus_buy_notional_usd",
                    "max_event_notional_usd",
                ]
            )

        frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True)
        frame["timestamp"] = frame["event_timestamp"].dt.floor("1h")
        frame["side_upper"] = frame["side"].astype(str).str.upper()
        frame["buy_notional_usd"] = frame["notional_usd"].where(frame["side_upper"].eq("BUY"), 0.0)
        frame["sell_notional_usd"] = frame["notional_usd"].where(frame["side_upper"].eq("SELL"), 0.0)
        frame["buy_count"] = frame["side_upper"].eq("BUY").astype(int)
        frame["sell_count"] = frame["side_upper"].eq("SELL").astype(int)

        grouped = frame.groupby("timestamp", sort=True)
        aggregate = (
            grouped.agg(
                event_count=("notional_usd", "size"),
                buy_count=("buy_count", "sum"),
                sell_count=("sell_count", "sum"),
                notional_usd=("notional_usd", "sum"),
                buy_notional_usd=("buy_notional_usd", "sum"),
                sell_notional_usd=("sell_notional_usd", "sum"),
                max_event_notional_usd=("notional_usd", "max"),
            )
            .reset_index()
            .sort_values("timestamp")
        )
        aggregate["sell_minus_buy_notional_usd"] = aggregate["sell_notional_usd"] - aggregate["buy_notional_usd"]
        return aggregate

    def persist_hourly_aggregates(
        self,
        db: Session,
        *,
        venue: str,
        symbol: str,
        interval: str,
        frame: pd.DataFrame,
        chunk_size: int = 250,
        commit: bool = True,
    ) -> int:
        if frame.empty:
            if commit:
                db.commit()
            return 0

        now = datetime.now(timezone.utc)
        records: list[dict[str, Any]] = []
        for row in frame.to_dict("records"):
            records.append(
                {
                    "venue": venue,
                    "symbol": symbol,
                    "interval": interval,
                    "timestamp": row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
                    "event_count": int(row["event_count"]),
                    "buy_count": int(row["buy_count"]),
                    "sell_count": int(row["sell_count"]),
                    "notional_usd": float(row["notional_usd"]),
                    "buy_notional_usd": float(row["buy_notional_usd"]),
                    "sell_notional_usd": float(row["sell_notional_usd"]),
                    "sell_minus_buy_notional_usd": float(row["sell_minus_buy_notional_usd"]),
                    "max_event_notional_usd": float(row["max_event_notional_usd"]),
                    "created_at": now,
                    "updated_at": now,
                }
            )

        persisted = 0
        for offset in range(0, len(records), chunk_size):
            chunk = records[offset : offset + chunk_size]
            stmt = sqlite_insert(DerivativesLiquidationHour).values(chunk)
            update_columns = {
                column.name: getattr(stmt.excluded, column.name)
                for column in DerivativesLiquidationHour.__table__.columns
                if column.name not in {"id", "created_at"}
            }
            result = db.execute(
                stmt.on_conflict_do_update(
                    index_elements=["venue", "symbol", "interval", "timestamp"],
                    set_=update_columns,
                )
            )
            persisted += int(result.rowcount or 0)

        if commit:
            db.commit()
        return persisted

    def rebuild_hourly_aggregates(
        self,
        db: Session,
        *,
        venue: str,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> int:
        start_hour = _floor_to_hour(start)
        end_hour = _floor_to_hour(end)
        if end <= start:
            raise ValueError("end must be later than start")

        rows = db.execute(
            select(
                DerivativesLiquidationEvent.event_timestamp,
                DerivativesLiquidationEvent.side,
                DerivativesLiquidationEvent.notional_usd,
            ).where(
                DerivativesLiquidationEvent.venue == venue,
                DerivativesLiquidationEvent.symbol == symbol,
                DerivativesLiquidationEvent.event_timestamp >= start_hour,
                DerivativesLiquidationEvent.event_timestamp < end_hour,
            )
        ).all()

        db.execute(
            delete(DerivativesLiquidationHour).where(
                DerivativesLiquidationHour.venue == venue,
                DerivativesLiquidationHour.symbol == symbol,
                DerivativesLiquidationHour.interval == interval,
                DerivativesLiquidationHour.timestamp >= start_hour,
                DerivativesLiquidationHour.timestamp < end_hour,
            )
        )

        frame = pd.DataFrame(
            [{"event_timestamp": row.event_timestamp, "side": row.side, "notional_usd": row.notional_usd} for row in rows]
        )
        aggregate = self.build_hourly_aggregate_frame(frame)
        return self.persist_hourly_aggregates(
            db,
            venue=venue,
            symbol=symbol,
            interval=interval,
            frame=aggregate,
            commit=True,
        )
