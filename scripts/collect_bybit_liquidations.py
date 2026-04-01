from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import SessionLocal, init_db
from app.core.exceptions import ExternalServiceError
from app.core.logging import configure_logging
from app.data.bybit_liquidation_stream_client import BybitLiquidationStreamClient
from app.services.derivatives_liquidation_service import DerivativesLiquidationService


DEFAULT_SUMMARY_PATH = ROOT / "artifacts" / "derivatives" / "bybit_liquidation_collector" / "last_run_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Bybit all-liquidation events and persist raw + hourly aggregates.")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="One or more linear symbols such as BTCUSDT.")
    parser.add_argument("--max-runtime-seconds", type=float, default=60.0, help="Stop after this many seconds.")
    parser.add_argument("--max-messages", type=int, default=None, help="Stop after this many liquidation websocket payloads.")
    parser.add_argument("--flush-size", type=int, default=50, help="Persist once this many normalized events are buffered.")
    parser.add_argument("--reconnect-delay-seconds", type=float, default=5.0, help="Wait before reconnecting after stream errors.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--skip-db", action="store_true", help="Do not persist to SQLite; useful for dry-run connectivity checks.")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def flush_pending(
    *,
    db,
    service: DerivativesLiquidationService,
    pending: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    if not pending:
        return

    inserted = service.persist_events(db, pending)
    stats["db_inserted_events"] += inserted

    per_symbol: dict[str, dict[str, datetime]] = defaultdict(lambda: {"start": None, "end": None})
    for row in pending:
        window = per_symbol[row["symbol"]]
        event_time = row["event_timestamp"]
        start = event_time.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
        if window["start"] is None or start < window["start"]:
            window["start"] = start
        if window["end"] is None or end > window["end"]:
            window["end"] = end

    for symbol, window in per_symbol.items():
        rebuilt = service.rebuild_hourly_aggregates(
            db,
            venue="bybit",
            symbol=symbol,
            interval="1h",
            start=window["start"],
            end=window["end"],
        )
        stats["hourly_rows_upserted"] += rebuilt

    stats["flushes"] += 1
    pending.clear()


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    logging.getLogger("websockets").setLevel(logging.WARNING)

    if not args.skip_db:
        init_db()

    client = BybitLiquidationStreamClient()
    service = DerivativesLiquidationService()
    pending: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "symbols": args.symbols,
        "requested_runtime_seconds": args.max_runtime_seconds,
        "flush_size": args.flush_size,
        "stream_sessions": 0,
        "acknowledged_sessions": 0,
        "raw_messages": 0,
        "liquidation_messages": 0,
        "normalized_events": 0,
        "db_inserted_events": 0,
        "hourly_rows_upserted": 0,
        "flushes": 0,
        "started_at": datetime.now(timezone.utc),
    }

    started = time.monotonic()
    db = SessionLocal() if not args.skip_db else None
    try:
        while True:
            elapsed = time.monotonic() - started
            remaining = args.max_runtime_seconds - elapsed
            if remaining <= 0:
                break

            stats["stream_sessions"] += 1

            def handle_payload(payload: dict[str, Any]) -> None:
                records = service.normalize_bybit_payload(payload)
                stats["normalized_events"] += len(records)
                pending.extend(records)
                if db is not None and len(pending) >= args.flush_size:
                    flush_pending(db=db, service=service, pending=pending, stats=stats)

            try:
                summary = client.stream(
                    symbols=args.symbols,
                    on_payload=handle_payload,
                    max_messages=args.max_messages,
                    max_runtime_seconds=remaining,
                )
            except ExternalServiceError as exc:
                logging.getLogger(__name__).warning("Liquidation stream session failed error=%s", exc)
                if remaining <= args.reconnect_delay_seconds:
                    break
                time.sleep(args.reconnect_delay_seconds)
                continue

            stats["raw_messages"] += int(summary["raw_messages"])
            stats["acknowledged_sessions"] += int(bool(summary["ack_received"]))
            stats["liquidation_messages"] += int(summary["liquidation_messages"])
            if args.max_messages is not None and stats["liquidation_messages"] >= args.max_messages:
                break
            if summary["runtime_seconds"] == 0:
                break

        if db is not None:
            flush_pending(db=db, service=service, pending=pending, stats=stats)
    finally:
        if db is not None:
            db.close()

    stats["completed_at"] = datetime.now(timezone.utc)
    stats["runtime_seconds"] = round(time.monotonic() - started, 4)
    write_summary(args.summary_path, stats)
    logging.getLogger(__name__).info("Liquidation collector summary=%s", stats)


if __name__ == "__main__":
    main()
