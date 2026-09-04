#!/usr/bin/env python3
"""Tool Server-compatible memory routes backed by a local OpenViking server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_CONTENT_LEVELS = frozenset({"read", "overview", "auto"})


@dataclass(frozen=True, slots=True)
class MemoryProxyConfig:
    enabled: bool = False
    openviking_target: str = ""
    openviking_url: str = "http://127.0.0.1:1933"
    openviking_api_key: str = ""
    openviking_account_id: str = "default"
    openviking_user_id: str = "default"
    timeout_seconds: float = 180.0
    event_log_file: Path | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not self.openviking_target.strip():
            raise ValueError("memory_proxy.openviking_target is required when enabled")
        if not self.openviking_url.strip():
            raise ValueError("memory_proxy.openviking_url is required when enabled")
        if not self.openviking_api_key.strip():
            raise ValueError("memory_proxy.openviking_api_key is required when enabled")
        if not self.openviking_account_id.strip():
            raise ValueError("memory_proxy.openviking_account_id is required when enabled")
        if not self.openviking_user_id.strip():
            raise ValueError("memory_proxy.openviking_user_id is required when enabled")
        if self.timeout_seconds <= 0:
            raise ValueError("memory_proxy.timeout_seconds must be > 0")


class LocalOpenVikingProxy:
    """Add local OV authentication and retain a credential-free traffic audit."""

    def __init__(
        self,
        config: MemoryProxyConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.call_count = 0
        self._event_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            base_url=config.openviking_url.rstrip("/"),
            headers={
                "X-API-Key": config.openviking_api_key,
                "X-OpenViking-Account": config.openviking_account_id,
                "X-OpenViking-User": config.openviking_user_id,
            },
            timeout=httpx.Timeout(config.timeout_seconds),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=40),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> JSONResponse:
        started_at = time.time()
        try:
            response = await self._client.request(method, path, params=params, json=body)
            try:
                content = response.json()
            except ValueError:
                result = _error(
                    "OPENVIKING_INVALID_RESPONSE",
                    f"local OpenViking returned non-JSON HTTP {response.status_code}",
                )
                status_code = 502
            else:
                if not isinstance(content, dict):
                    result = _error(
                        "OPENVIKING_INVALID_RESPONSE",
                        "local OpenViking returned a non-object JSON response",
                    )
                    status_code = 502
                else:
                    result = content
                    status_code = response.status_code
        except httpx.TimeoutException:
            result = _error(
                "OPENVIKING_TIMEOUT",
                f"local OpenViking request timed out: {method} {path}",
            )
            status_code = 504
        except httpx.HTTPError as exc:
            result = _error(
                "OPENVIKING_UNAVAILABLE",
                f"local OpenViking request failed: {method} {path}: {type(exc).__name__}",
            )
            status_code = 502

        self.call_count += 1
        await self._record_event(
            method=method,
            path=path,
            params=params,
            body=body,
            status_code=status_code,
            duration_ms=round((time.time() - started_at) * 1000, 3),
        )
        return JSONResponse(status_code=status_code, content=result)

    async def _record_event(
        self,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        body: dict[str, Any] | None,
        status_code: int,
        duration_ms: float,
    ) -> None:
        event_path = self.config.event_log_file
        if event_path is None:
            return
        fingerprint_input = json.dumps(
            {"params": params or {}, "body": body or {}},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        event = {
            "observed_at": time.time(),
            "openviking_target": self.config.openviking_target,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "request_keys": sorted((body or {}).keys()),
            "query_keys": sorted((params or {}).keys()),
            "request_sha256": hashlib.sha256(fingerprint_input).hexdigest(),
        }
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        async with self._event_lock:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with event_path.open("a", encoding="utf-8") as stream:
                stream.write(line)


def install_memory_proxy(
    app: FastAPI,
    config: MemoryProxyConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> LocalOpenVikingProxy | None:
    """Install the four memory endpoints used by Vaka Tool Server."""

    if not config.enabled:
        app.state.memory_proxy = None
        return None

    proxy = LocalOpenVikingProxy(config, transport=transport)
    app.state.memory_proxy = proxy
    app.router.add_event_handler("shutdown", proxy.close)

    @app.post("/api/v1/search/find")
    async def search_find(payload: dict[str, Any]) -> JSONResponse:
        return await proxy.request("POST", "/api/v1/search/find", body=payload)

    @app.post("/api/v1/search/search")
    async def search_search(payload: dict[str, Any]) -> JSONResponse:
        return await proxy.request("POST", "/api/v1/search/search", body=payload)

    @app.get("/api/v1/content/{level}")
    async def content(level: str, request: Request) -> JSONResponse:
        if level not in _CONTENT_LEVELS:
            return JSONResponse(
                status_code=400,
                content=_error(
                    "BAD_REQUEST",
                    f"unsupported content level {level!r}; expected read, overview, or auto",
                ),
            )
        params = dict(request.query_params)
        if level == "auto":
            uri = str(params.get("uri") or "").strip()
            if not uri:
                return JSONResponse(
                    status_code=400,
                    content=_error("BAD_REQUEST", "query parameter 'uri' is required"),
                )
            stat_response = await proxy.request(
                "GET",
                "/api/v1/fs/stat",
                params={"uri": uri},
            )
            if stat_response.status_code >= 400:
                return stat_response
            try:
                stat_body = json.loads(bytes(stat_response.body))
                stat_result = stat_body["result"]
                is_dir = bool(stat_result.get("isDir", stat_result.get("is_dir", False)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return JSONResponse(
                    status_code=502,
                    content=_error(
                        "OPENVIKING_INVALID_RESPONSE",
                        "local OpenViking stat response has no result object",
                    ),
                )
            level = "overview" if is_dir else "read"
        if level == "read":
            params.setdefault("raw", "false")
        return await proxy.request("GET", f"/api/v1/content/{level}", params=params)

    @app.get("/api/v1/fs/stat")
    async def stat(request: Request) -> JSONResponse:
        return await proxy.request(
            "GET",
            "/api/v1/fs/stat",
            params=dict(request.query_params),
        )

    return proxy


def _error(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": {"code": code, "message": message}}
