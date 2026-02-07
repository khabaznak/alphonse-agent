from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, Body, HTTPException, Header
from fastapi.responses import StreamingResponse

from alphonse.infrastructure.api_gateway import gateway
from alphonse.infrastructure.web_event_hub import web_event_hub
from alphonse.agent.lan.api import router as lan_router

app = FastAPI(title="Alphonse API", version="0.1.0")
app.include_router(lan_router, prefix="/lan")


@app.get("/agent/status")
def agent_status(x_alphonse_api_token: str | None = Header(default=None)) -> dict[str, object]:
    _assert_api_token(x_alphonse_api_token)
    signal = gateway.build_signal(
        "api.status_requested",
        {"api_token": x_alphonse_api_token},
        None,
    )
    response = gateway.emit_and_wait(signal, timeout=5.0)
    if response is None:
        raise HTTPException(status_code=503, detail="API gateway unavailable")
    return response


@app.post("/agent/message")
def agent_message(
    payload: dict[str, Any] = Body(...),
    x_alphonse_api_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    text = str(payload.get("text", "")).strip()
    signal = gateway.build_signal(
        "api.message_received",
        {
            "text": text,
            "args": payload.get("args") or {},
            "user_id": payload.get("user_id"),
            "channel": payload.get("channel") or "webui",
            "metadata": payload.get("metadata") or {},
            "timestamp": payload.get("timestamp") or time.time(),
            "api_token": x_alphonse_api_token,
        },
        _as_optional_str(payload.get("correlation_id")),
    )
    response = gateway.emit_and_wait(signal, timeout=5.0)
    if response is None:
        raise HTTPException(status_code=503, detail="API gateway unavailable")
    return response


@app.get("/agent/timed-signals")
def timed_signals(limit: int = 200, x_alphonse_api_token: str | None = Header(default=None)) -> dict[str, Any]:
    _assert_api_token(x_alphonse_api_token)
    signal = gateway.build_signal(
        "api.timed_signals_requested",
        {"limit": limit, "api_token": x_alphonse_api_token},
        None,
    )
    response = gateway.emit_and_wait(signal, timeout=5.0)
    if response is None:
        raise HTTPException(status_code=503, detail="API gateway unavailable")
    return response


@app.get("/agent/events")
def agent_events(
    channel_target: str = "webui",
    x_alphonse_api_token: str | None = Header(default=None),
) -> StreamingResponse:
    _assert_api_token(x_alphonse_api_token)
    subscriber_id = web_event_hub.subscribe(channel_target)

    def _stream():
        try:
            while True:
                event = web_event_hub.next_event(subscriber_id, timeout=15.0)
                if event is None:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            web_event_hub.unsubscribe(subscriber_id)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _assert_api_token(provided: str | None) -> None:
    expected = os.getenv("ALPHONSE_API_TOKEN")
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API token")
