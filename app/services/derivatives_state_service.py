from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.data.bybit_derivatives_client import BybitDerivativesClient
from app.models.derivatives_state_bar import DerivativesStateBar


logger = logging.getLogger(__name__)

HOUR = timedelta(hours=1)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _floor_to_hour(value: datetime) -> datetime:
    value = _to_utc(value)
    return value.replace(minute=0, second=0, microsecond=0)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=max(24, window // 4)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(24, window // 4)).std()
    safe_std = rolling_std.where(rolling_std.ne(0.0))
    return ((series - rolling_mean) / safe_std).astype("float64")


class DerivativesStateService:
    def __init__(self, client: BybitDerivativesClient) -> None:
        self.client = client

    def build_hourly_panel(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        start_hour = _floor_to_hour(start)
        end_hour = _floor_to_hour(end)
        if end_hour <= start_hour:
            raise ValueError("end must be later than start by at least one hour")

        funding = self.client.fetch_funding_history(symbol=symbol, start=start_hour - timedelta(hours=24), end=end_hour)
        open_interest = self.client.fetch_open_interest_history(symbol=symbol, start=start_hour, end=end_hour)
        mark = self.client.fetch_mark_price_klines(symbol=symbol, start=start_hour, end=end_hour)
        index = self.client.fetch_index_price_klines(symbol=symbol, start=start_hour, end=end_hour)
        premium = self.client.fetch_premium_index_klines(symbol=symbol, start=start_hour, end=end_hour)

        frame = pd.DataFrame({"timestamp": pd.date_range(start=start_hour, end=end_hour, freq="1h", inclusive="left", tz="UTC")})
        for dataset in (mark, index, premium, open_interest, funding):
            frame = frame.merge(dataset, on="timestamp", how="left")

        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame["funding_rate"] = frame["funding_rate_event"].ffill()
        frame["open_interest_notional_proxy_usd"] = frame["open_interest"] * frame["index_close"]
        frame["open_interest_change_1h_pct"] = frame["open_interest"].pct_change() * 100.0
        frame["basis_proxy_bps"] = frame["premium_close"] * 10000.0
        frame["mark_index_spread"] = frame["mark_close"] - frame["index_close"]
        frame["mark_index_spread_bps"] = ((frame["mark_close"] / frame["index_close"]) - 1.0) * 10000.0

        missing_core = frame[["mark_close", "index_close", "open_interest"]].isna().any(axis=1).sum()
        if missing_core:
            logger.warning("Derivatives panel has %s rows with missing core fields symbol=%s", int(missing_core), symbol)

        return frame

    def build_research_table(self, panel: pd.DataFrame) -> pd.DataFrame:
        frame = panel.copy().sort_values("timestamp").reset_index(drop=True)
        frame["funding_rate_z_7d"] = _rolling_zscore(frame["funding_rate"], window=24 * 7)
        frame["basis_proxy_bps_z_7d"] = _rolling_zscore(frame["basis_proxy_bps"], window=24 * 7)
        frame["mark_index_spread_bps_z_7d"] = _rolling_zscore(frame["mark_index_spread_bps"], window=24 * 7)
        frame["open_interest_change_z_7d"] = _rolling_zscore(frame["open_interest_change_1h_pct"], window=24 * 7)
        frame["open_interest_notional_change_1h_pct"] = frame["open_interest_notional_proxy_usd"].pct_change() * 100.0
        frame["open_interest_notional_change_z_7d"] = _rolling_zscore(
            frame["open_interest_notional_change_1h_pct"],
            window=24 * 7,
        )

        for horizon in (1, 4, 24):
            future_index = frame["index_close"].shift(-horizon)
            frame[f"forward_index_return_bps_{horizon}h"] = ((future_index / frame["index_close"]) - 1.0) * 10000.0
            frame[f"forward_abs_index_return_bps_{horizon}h"] = frame[f"forward_index_return_bps_{horizon}h"].abs()

        return frame

    def persist_hourly_panel(
        self,
        db: Session,
        *,
        venue: str,
        symbol: str,
        interval: str,
        panel: pd.DataFrame,
        chunk_size: int = 250,
    ) -> int:
        columns = [
            "timestamp",
            "mark_close",
            "index_close",
            "premium_close",
            "funding_rate_event",
            "funding_rate",
            "open_interest",
            "open_interest_notional_proxy_usd",
            "open_interest_change_1h_pct",
            "basis_proxy_bps",
            "mark_index_spread",
            "mark_index_spread_bps",
        ]
        records: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for row in panel[columns].to_dict("records"):
            normalized = {
                key: (value.to_pydatetime() if hasattr(value, "to_pydatetime") else value)
                for key, value in row.items()
            }
            normalized["venue"] = venue
            normalized["symbol"] = symbol
            normalized["interval"] = interval
            normalized["created_at"] = now
            normalized["updated_at"] = now
            records.append(normalized)

        if not records:
            return 0

        for offset in range(0, len(records), chunk_size):
            chunk = records[offset : offset + chunk_size]
            stmt = sqlite_insert(DerivativesStateBar).values(chunk)
            update_columns = {
                column.name: getattr(stmt.excluded, column.name)
                for column in DerivativesStateBar.__table__.columns
                if column.name not in {"id", "created_at"}
            }
            db.execute(
                stmt.on_conflict_do_update(
                    index_elements=["venue", "symbol", "interval", "timestamp"],
                    set_=update_columns,
                )
            )
        db.commit()
        return len(records)
