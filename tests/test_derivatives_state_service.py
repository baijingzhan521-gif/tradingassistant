from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from app.services.derivatives_state_service import DerivativesStateService


class FakeBybitClient:
    def fetch_funding_history(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                        datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
                    ],
                    utc=True,
                ),
                "funding_rate_event": [0.0001, 0.0002],
            }
        )

    def fetch_open_interest_history(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        hours = pd.date_range(start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), periods=80, freq="1h", tz="UTC")
        return pd.DataFrame({"timestamp": hours, "open_interest": [1000 + idx * 10 for idx in range(len(hours))]})

    def fetch_mark_price_klines(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        hours = pd.date_range(start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), periods=80, freq="1h", tz="UTC")
        return pd.DataFrame({"timestamp": hours, "mark_close": [100 + idx for idx in range(len(hours))]})

    def fetch_index_price_klines(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        hours = pd.date_range(start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), periods=80, freq="1h", tz="UTC")
        return pd.DataFrame({"timestamp": hours, "index_close": [99.5 + idx for idx in range(len(hours))]})

    def fetch_premium_index_klines(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        hours = pd.date_range(start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), periods=80, freq="1h", tz="UTC")
        premiums = [0.0001 if idx < 40 else 0.0002 for idx in range(len(hours))]
        return pd.DataFrame({"timestamp": hours, "premium_close": premiums})


def test_build_hourly_panel_aligns_and_derives_features() -> None:
    service = DerivativesStateService(FakeBybitClient())
    panel = service.build_hourly_panel(
        symbol="BTCUSDT",
        start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, 8, 0, tzinfo=timezone.utc),
    )

    assert len(panel) == 80
    assert panel.loc[0, "funding_rate"] == 0.0001
    assert panel.loc[9, "funding_rate"] == 0.0002
    assert round(panel.loc[0, "basis_proxy_bps"], 4) == 1.0
    assert round(panel.loc[0, "mark_index_spread_bps"], 4) == round(((100.0 / 99.5) - 1.0) * 10000.0, 4)
    assert round(panel.loc[1, "open_interest_change_1h_pct"], 6) == 1.0


def test_build_research_table_adds_labels_and_zscores() -> None:
    service = DerivativesStateService(FakeBybitClient())
    panel = service.build_hourly_panel(
        symbol="BTCUSDT",
        start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, 8, 0, tzinfo=timezone.utc),
    )
    research = service.build_research_table(panel)

    assert "funding_rate_z_7d" in research.columns
    assert "open_interest_change_z_7d" in research.columns
    assert "forward_index_return_bps_24h" in research.columns
    expected_1h = ((research.loc[1, "index_close"] / research.loc[0, "index_close"]) - 1.0) * 10000.0
    assert round(research.loc[0, "forward_index_return_bps_1h"], 6) == round(expected_1h, 6)
    assert research["funding_rate_z_7d"].notna().sum() > 0
