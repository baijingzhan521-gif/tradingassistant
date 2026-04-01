from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_workspace_page_is_available_on_root_and_named_route() -> None:
    app = create_app()

    with TestClient(app) as client:
        root_response = client.get("/")
        named_response = client.get("/workspace")

    assert root_response.status_code == 200
    assert named_response.status_code == 200
    assert "text/html" in root_response.headers["content-type"]
    assert "交易分析助手" in root_response.text
    assert "一次分析会同时生成主线波段和日内参考两套结果" in root_response.text
    assert "开始分析" in root_response.text
    assert "/analyze" in root_response.text
    assert "/analyses?limit=12" in root_response.text
    assert "/analysis/" in root_response.text
    assert "整体建议" in root_response.text
    assert "双窗口结果" in root_response.text
    assert "日内交易" in root_response.text
    assert "波段交易" in root_response.text
    assert "1H / 15m / 3m" in root_response.text
    assert "1d / 4h / 1h" in root_response.text
    assert "结构图" in root_response.text
    assert "开始对比" in root_response.text
    assert "BTC 回测交易" in root_response.text
    assert "/workspace/backtest-trades/btc-best" in root_response.text
    assert "展开详细诊断" in root_response.text
    assert root_response.text == named_response.text
