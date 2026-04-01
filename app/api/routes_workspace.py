from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["workspace"])

_WORKSPACE_HTML_PATH = Path(__file__).resolve().parents[1] / "static" / "workspace.html"


@lru_cache(maxsize=1)
def _load_workspace_html() -> str:
    return _WORKSPACE_HTML_PATH.read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
@router.get("/workspace", response_class=HTMLResponse)
def workspace_page() -> HTMLResponse:
    return HTMLResponse(content=_load_workspace_html())
