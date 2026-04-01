from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from app.utils.timeframes import TIMEFRAME_TO_MINUTES, get_strategy_fetch_timeframes
from scripts.backtest_snapshot_io import save_enriched_history_snapshot


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "snapshots" / "latest"
DEFAULT_PROFILES = [
    "intraday_mtf_v2",
    "intraday_mtf_v2_pullback_075_v1",
    "intraday_mtf_v2_trend70_v1",
    "intraday_mtf_v2_cooldown10_v1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable enriched-history snapshot for offline backtest replay.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None, help="Defaults to now (UTC).")
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "artifacts" / "backtests" / "cache"),
        help="Directory of cached OHLCV csv files (cache-only snapshot build).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def make_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = {
        "exit_profile": "long_scaled1_3_short_fixed1_5",
        "take_profit_mode": "scaled",
        "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
        "swing_detection_mode": "confirmed",
        "cache_dir": "artifacts/backtests/cache",
    }
    if assumption_overrides:
        assumptions.update(assumption_overrides)
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**assumptions),
    )


def summarize_history(enriched_history: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    lines: list[str] = []
    for profile in sorted(enriched_history.keys()):
        frames = enriched_history[profile]
        parts: list[str] = []
        for timeframe in sorted(frames.keys()):
            frame = frames[timeframe]
            parts.append(f"{timeframe}:{len(frame)}")
        lines.append(f"{profile} -> " + ", ".join(parts))
    return lines


def _cache_symbol_key(symbol: str) -> str:
    return symbol.lower().replace("/", "_").replace(":", "_")


def _parse_cache_range(path: Path, *, symbol_key: str, timeframe: str) -> tuple[datetime, datetime] | None:
    pattern = re.compile(rf"^{re.escape(symbol_key)}_{re.escape(timeframe)}_(\d{{8}})_(\d{{8}})\.csv$")
    match = pattern.match(path.name)
    if not match:
        return None
    start = datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(match.group(2), "%Y%m%d").replace(tzinfo=timezone.utc)
    return start, end


def _pick_cache_file(
    *,
    cache_dir: Path,
    symbol_key: str,
    timeframe: str,
    required_start: datetime,
    required_end: datetime,
) -> Path:
    candidates: list[tuple[Path, datetime, datetime]] = []
    for path in cache_dir.glob(f"{symbol_key}_{timeframe}_*.csv"):
        parsed = _parse_cache_range(path, symbol_key=symbol_key, timeframe=timeframe)
        if parsed is None:
            continue
        start, end = parsed
        if start <= required_start and end >= required_end:
            candidates.append((path, start, end))
    if not candidates:
        raise FileNotFoundError(
            f"No cache file covers {required_start.date().isoformat()}->{required_end.date().isoformat()} "
            f"for {symbol_key} {timeframe} under {cache_dir}"
        )
    # Prefer the narrowest valid span to reduce IO.
    candidates.sort(key=lambda item: (item[2] - item[1], item[1]))
    return candidates[0][0]


def _load_raw_frames_from_cache(
    *,
    cache_dir: Path,
    symbol: str,
    profile: str,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    symbol_key = _cache_symbol_key(symbol)
    raw: dict[str, pd.DataFrame] = {}
    for timeframe in get_strategy_fetch_timeframes(profile):
        warmup_minutes = TIMEFRAME_TO_MINUTES[timeframe] * 300
        required_start = start - timedelta(minutes=warmup_minutes)
        path = _pick_cache_file(
            cache_dir=cache_dir,
            symbol_key=symbol_key,
            timeframe=timeframe,
            required_start=required_start,
            required_end=end,
        )
        frame = pd.read_csv(path)
        if "timestamp" not in frame.columns:
            raise ValueError(f"cache file missing timestamp column: {path}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame[(frame["timestamp"] >= pd.Timestamp(required_start)) & (frame["timestamp"] <= pd.Timestamp(end))].copy()
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"cache file produced empty frame after trim: {path}")
        raw[timeframe] = frame
    return raw


def main() -> None:
    args = parse_args()
    configure_logging()

    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    if end <= start:
        raise ValueError("end must be later than start")

    profiles = list(dict.fromkeys(args.profiles))
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")
    service = make_service()
    enriched_history: dict[str, dict[str, pd.DataFrame]] = {}
    for profile in profiles:
        raw_frames = _load_raw_frames_from_cache(
            cache_dir=cache_dir,
            symbol=args.symbol,
            profile=profile,
            start=start,
            end=end,
        )
        enriched_history[profile] = service._prepare_enriched_frames(strategy_profile=profile, frames=raw_frames)

    output_dir = Path(args.output_dir)
    save_enriched_history_snapshot(
        snapshot_dir=output_dir,
        symbol=args.symbol,
        exchange=args.exchange,
        market_type=args.market_type,
        start=start,
        end=end,
        enriched_history=enriched_history,
    )

    print(f"saved snapshot to {output_dir}")
    for line in summarize_history(enriched_history):
        print(line)


if __name__ == "__main__":
    main()
