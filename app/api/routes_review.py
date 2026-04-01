from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["review"])

_REVIEW_HTML_PATH = Path(__file__).resolve().parents[1] / "static" / "review.html"


@lru_cache(maxsize=1)
def _load_review_html() -> str:
    return _REVIEW_HTML_PATH.read_text(encoding="utf-8")


@router.get("/review", response_class=HTMLResponse)
def review_page() -> HTMLResponse:
    return HTMLResponse(content=_load_review_html())
