from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from scripts.build_backtest_snapshot import _parse_cache_range, _pick_cache_file


def test_parse_cache_range_success() -> None:
    path = Path("btc_usdt_usdt_1h_20191219_20260331.csv")
    parsed = _parse_cache_range(path, symbol_key="btc_usdt_usdt", timeframe="1h")
    assert parsed is not None
    start, end = parsed
    assert start == datetime(2019, 12, 19, tzinfo=UTC)
    assert end == datetime(2026, 3, 31, tzinfo=UTC)


def test_pick_cache_file_prefers_narrower_valid_span(tmp_path) -> None:
    (tmp_path / "btc_usdt_usdt_1h_20190101_20260331.csv").write_text("timestamp\n", encoding="utf-8")
    (tmp_path / "btc_usdt_usdt_1h_20191219_20260331.csv").write_text("timestamp\n", encoding="utf-8")
    chosen = _pick_cache_file(
        cache_dir=tmp_path,
        symbol_key="btc_usdt_usdt",
        timeframe="1h",
        required_start=datetime(2020, 1, 1, tzinfo=UTC),
        required_end=datetime(2026, 3, 31, tzinfo=UTC),
    )
    assert chosen.name == "btc_usdt_usdt_1h_20191219_20260331.csv"
