from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _profile_dir(snapshot_dir: Path, profile: str) -> Path:
    return snapshot_dir / profile


def _timeframe_path(snapshot_dir: Path, profile: str, timeframe: str) -> Path:
    return _profile_dir(snapshot_dir, profile) / f"{timeframe}.csv"


def save_enriched_history_snapshot(
    *,
    snapshot_dir: Path,
    symbol: str,
    exchange: str,
    market_type: str,
    start: datetime,
    end: datetime,
    enriched_history: dict[str, dict[str, pd.DataFrame]],
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    manifest_profiles: dict[str, list[str]] = {}
    for profile, frames in enriched_history.items():
        profile_path = _profile_dir(snapshot_dir, profile)
        profile_path.mkdir(parents=True, exist_ok=True)
        manifest_profiles[profile] = sorted(frames.keys())
        for timeframe, frame in frames.items():
            output_path = _timeframe_path(snapshot_dir, profile, timeframe)
            frame.to_csv(output_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "exchange": exchange,
        "market_type": market_type,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "profiles": manifest_profiles,
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def load_enriched_history_snapshot(
    *,
    snapshot_dir: Path,
    profiles: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"snapshot dir not found: {snapshot_dir}")

    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for profile in profiles:
        profile_path = _profile_dir(snapshot_dir, profile)
        if not profile_path.exists():
            raise FileNotFoundError(f"snapshot profile not found: {profile_path}")
        timeframe_files = sorted(profile_path.glob("*.csv"))
        if not timeframe_files:
            raise FileNotFoundError(f"snapshot profile has no timeframe csv: {profile_path}")

        frames: dict[str, pd.DataFrame] = {}
        for file_path in timeframe_files:
            timeframe = file_path.stem
            frame = pd.read_csv(file_path)
            if "timestamp" not in frame.columns:
                raise ValueError(f"snapshot frame missing timestamp column: {file_path}")
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.sort_values("timestamp").reset_index(drop=True)
            frames[timeframe] = frame
        loaded[profile] = frames

    return loaded


def snapshot_metadata(snapshot_dir: Path) -> dict[str, Any] | None:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))
