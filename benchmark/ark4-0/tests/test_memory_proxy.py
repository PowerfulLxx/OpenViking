from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from memory_proxy import MemoryProxyConfig, install_memory_proxy


@pytest.mark.asyncio
async def test_memory_proxy_adds_local_api_key_and_records_safe_event(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["X-API-Key"] == "local-secret"
        assert request.headers["X-OpenViking-Account"] == "local-account"
        assert request.headers["X-OpenViking-User"] == "local-user"
        return httpx.Response(
            200,
            json={"result": {"total": 0, "memories": []}},
        )

    app = FastAPI()
    event_log = tmp_path / "memory-events.jsonl"
    proxy = install_memory_proxy(
        app,
        MemoryProxyConfig(
            enabled=True,
            openviking_target="ov-ark-test",
            openviking_url="http://openviking.test",
            openviking_api_key="local-secret",
            openviking_account_id="local-account",
            openviking_user_id="local-user",
            event_log_file=event_log,
        ),
        transport=httpx.MockTransport(handler),
    )
    assert proxy is not None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://adapter.test",
        ) as client:
            response = await client.post(
                "/api/v1/search/find",
                json={"query": "private query", "limit": 1},
            )
    finally:
        await proxy.close()

    assert response.status_code == 200
    assert requests[0].url.path == "/api/v1/search/find"
    event = json.loads(event_log.read_text(encoding="utf-8"))
    assert event["openviking_target"] == "ov-ark-test"
    assert event["request_keys"] == ["limit", "query"]
    assert "private query" not in event_log.read_text(encoding="utf-8")
    assert "local-secret" not in event_log.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_memory_proxy_resolves_auto_content_level(tmp_path: Path) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(200, json={"result": {"isDir": False}})
        assert request.url.params["raw"] == "false"
        return httpx.Response(200, json={"result": {"content": "ok"}})

    app = FastAPI()
    proxy = install_memory_proxy(
        app,
        MemoryProxyConfig(
            enabled=True,
            openviking_target="ov-ark-test",
            openviking_url="http://openviking.test",
            openviking_api_key="local-secret",
            event_log_file=tmp_path / "events.jsonl",
        ),
        transport=httpx.MockTransport(handler),
    )
    assert proxy is not None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://adapter.test",
        ) as client:
            response = await client.get(
                "/api/v1/content/auto",
                params={"uri": "viking://user/memories/example.md"},
            )
    finally:
        await proxy.close()

    assert response.status_code == 200
    assert paths == ["/api/v1/fs/stat", "/api/v1/content/read"]
