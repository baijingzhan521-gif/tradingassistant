from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from scripts.backtest_snapshot_io import load_enriched_history_snapshot, save_enriched_history_snapshot


def _frame(values: list[tuple[str, float]]) -> pd.DataFrame:
    rows = []
    for ts, close in values:
        rows.append(
            {
                "timestamp": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_save_and_load_snapshot_roundtrip(tmp_path) -> None:
    snapshot_dir = tmp_path / "snap"
    enriched = {
        "profile_a": {
            "1h": _frame([("2026-01-01T00:00:00Z", 1.0), ("2026-01-01T01:00:00Z", 2.0)]),
            "15m": _frame([("2026-01-01T00:00:00Z", 1.1)]),
        }
    }
    save_enriched_history_snapshot(
        snapshot_dir=snapshot_dir,
        symbol="BTC/USDT:USDT",
        exchange="binance",
        market_type="perpetual",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        enriched_history=enriched,
    )

    loaded = load_enriched_history_snapshot(snapshot_dir=snapshot_dir, profiles=["profile_a"])
    assert set(loaded.keys()) == {"profile_a"}
    assert set(loaded["profile_a"].keys()) == {"1h", "15m"}
    assert len(loaded["profile_a"]["1h"]) == 2
    assert str(loaded["profile_a"]["1h"]["timestamp"].dtype).startswith("datetime64[ns, UTC]")


def test_load_snapshot_missing_profile_raises(tmp_path) -> None:
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    try:
        load_enriched_history_snapshot(snapshot_dir=snapshot_dir, profiles=["missing_profile"])
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
