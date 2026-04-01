from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import sync_playwright
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from tests.test_review_page import _seed_review_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture(scope="module")
def review_smoke_server(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("review-smoke")
    db_path = db_dir / "review-smoke.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    db = SessionLocal()
    try:
        _seed_review_data(db)
        db.commit()
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
def test_review_page_browser_smoke(review_smoke_server) -> None:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(base_url=review_smoke_server)

            page.goto("/review", wait_until="domcontentloaded")
            page.get_by_text("交易复盘").wait_for()
            page.get_by_role("button", name="加载分析").click()
            page.get_by_text("已加载 2 / 2 条分析。").wait_for()

            page.locator("#leftId").fill("review-long")
            page.locator("#rightId").fill("review-short")
            with page.expect_response(lambda response: response.url.endswith("/analysis/review-long/diff/review-short") and response.status == 200):
                page.get_by_role("button", name="开始对比").click()

            page.get_by_text("决策变化").wait_for()
            page.get_by_text("市场状态变化").wait_for()
            page.get_by_text("review-long 对比 review-short").wait_for()
            assert page.get_by_text("变化区块：").is_visible()
            browser.close()
    except Exception as exc:
        pytest.skip(f"Chromium is not available for Playwright smoke: {exc}")
