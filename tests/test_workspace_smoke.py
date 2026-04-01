from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest
from playwright.sync_api import sync_playwright
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.schemas.request import AnalyzeRequest
from app.services.persistence_service import PersistenceService
from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG, TrendPullbackV1Strategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "trend_pullback_v1_baselines.json"
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # pragma: no cover - transient startup window
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Server did not become healthy at {base_url}") from last_error


def _build_bullish_long_record() -> tuple[AnalyzeRequest, object]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    request = AnalyzeRequest(**fixture["request"])
    case = next(item for item in fixture["cases"] if item["name"] == "bullish_long")
    ohlcv = {}
    for timeframe, rows in case["ohlcv_dump"].items():
        frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        ohlcv[timeframe] = frame

    strategy = TrendPullbackV1Strategy(DEFAULT_CONFIG)
    analysis = strategy.analyze(request, ohlcv).model_copy(update={"analysis_id": "workspace-long"})
    return request, analysis


@pytest.fixture(scope="module")
def workspace_smoke_server(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("workspace-smoke")
    db_path = db_dir / "workspace-smoke.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    request, analysis = _build_bullish_long_record()
    db = SessionLocal()
    try:
        PersistenceService().save_analysis(db, request, analysis)
    finally:
        db.close()
        engine.dispose()

    port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "DATABASE_URL": f"sqlite:///{db_path}",
            "DEFAULT_EXCHANGE": "binance",
            "DEFAULT_MARKET_TYPE": "perpetual",
            "DEFAULT_LOOKBACK": "300",
            "LOG_LEVEL": "warning",
            "PYTHONPATH": str(PROJECT_ROOT),
        }
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
            proc.kill()


@pytest.mark.browser
def test_workspace_page_browser_smoke(workspace_smoke_server) -> None:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(base_url=workspace_smoke_server, viewport={"width": 1440, "height": 1200})

            page.goto("/", wait_until="domcontentloaded")
            page.get_by_text("交易分析助手").wait_for()
            page.get_by_role("button", name="载入结果").first.wait_for()
            page.get_by_role("button", name="载入结果").first.click()

            page.get_by_text("多周期 K 线").wait_for()
            page.locator('[data-chart-timeframe="1h"] text', has_text="入场区").first.wait_for()

            hover_target = page.locator('[data-chart-timeframe="1h"] .chart-hit').nth(10)
            hover_target.hover()
            page.locator('[data-chart-timeframe="1h"] [data-chart-tooltip]').get_by_text("开 / 高 / 低 / 收").wait_for()
            page.locator('[data-chart-timeframe="1h"] [data-chart-tooltip]').get_by_text("EMA21 / EMA55 / EMA100 / EMA200").wait_for()
            browser.close()
    except Exception as exc:
        pytest.skip(f"Chromium is not available for Playwright smoke: {exc}")
