from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from fastapi import APIRouter, Body, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from alphonse.agent.lan.store import (
    DEFAULT_ALLOWED_SCOPES,
    consume_pairing_code,
    generate_pairing_code,
    get_latest_last_seen,
    get_paired_device,
    is_paired_device,
    register_device,
    update_device_last_seen,
    update_device_status,
)
from alphonse.agent.lan.ws import LanConnectionManager


router = APIRouter(tags=["lan"])
_connections = LanConnectionManager()


class PairRequest(BaseModel):
    pair_code: str
    device_id: str
    device_name: str | None = None


class PairingCodeResponse(BaseModel):
    pair_code: str
    expires_at: str


class CommandRequest(BaseModel):
    type: str = Field(..., description="Command type")
    payload: dict[str, Any] = Field(default_factory=dict)


class StatusPayload(BaseModel):
    timestamp: str | None = None
    activity: str | None = None
    confidence: int | None = None
    battery_percent: int | None = None
    dnd_enabled: bool | None = None


@router.post("/pairing-codes", response_model=PairingCodeResponse)
def create_pairing_code() -> PairingCodeResponse:
    ttl_minutes = _pairing_ttl_minutes()
    code = generate_pairing_code(ttl_minutes=ttl_minutes)
    return PairingCodeResponse(pair_code=code.code, expires_at=code.expires_at.isoformat())


@router.post("/pair")
def pair_device(payload: PairRequest) -> dict[str, Any]:
    if get_paired_device(payload.device_id):
        raise HTTPException(status_code=409, detail="Device already paired")
    if not consume_pairing_code(payload.pair_code):
        raise HTTPException(status_code=401, detail="Invalid or expired pairing code")
    device = register_device(
        device_id=payload.device_id,
        device_name=payload.device_name,
        allowed_scopes=list(DEFAULT_ALLOWED_SCOPES),
    )
    return {
        "status": "paired",
        "device_id": device.device_id,
        "device_name": device.device_name,
        "paired_at": device.paired_at.isoformat(),
        "allowed_scopes": device.allowed_scopes,
    }


@router.get("/status")
def lan_status() -> dict[str, Any]:
    last_seen = get_latest_last_seen()
    return {
        "status": "ok",
        "alphonse_online": True,
        "version": os.getenv("ALPHONSE_VERSION", "dev"),
        "last_seen_device": last_seen.isoformat() if last_seen else None,
    }


@router.post("/command")
async def handle_command(
    payload: CommandRequest,
    x_alphonse_device_id: str | None = Header(default=None),
) -> dict[str, Any]:
    device_id = _require_device_id(x_alphonse_device_id)
    if not is_paired_device(device_id):
        raise HTTPException(status_code=403, detail="Device not paired")
    update_device_last_seen(device_id)
    if payload.type != "request_status":
        raise HTTPException(status_code=400, detail="Unsupported command type")
    await _connections.send_ack(device_id, "request_status")
    device = get_paired_device(device_id)
    if device and device.last_status:
        await _connections.send(
            device_id,
            {
                "type": "status",
                "timestamp": _utcnow_iso(),
                "payload": device.last_status,
            },
        )
    return {"status": "accepted"}


@router.post("/status")
async def handle_status(
    payload: StatusPayload = Body(...),
    x_alphonse_device_id: str | None = Header(default=None),
) -> dict[str, Any]:
    device_id = _require_device_id(x_alphonse_device_id)
    if not is_paired_device(device_id):
        raise HTTPException(status_code=403, detail="Device not paired")
    update_device_status(device_id, payload.model_dump())
    await _connections.send_ack(device_id, "send_status")
    return {"status": "ok"}


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    device_id = websocket.query_params.get("device_id") or websocket.headers.get(
        "x-alphonse-device-id"
    )
    if not device_id or not is_paired_device(device_id):
        await websocket.close(code=4401)
        return
    await _connections.connect(device_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections.disconnect(device_id)


def _pairing_ttl_minutes() -> int:
    raw = os.getenv("ALPHONSE_PAIRING_TTL_MINUTES", "15")
    try:
        ttl = int(raw)
    except ValueError:
        return 15
    return max(ttl, 1)


def _require_device_id(device_id: str | None) -> str:
    if not device_id:
        raise HTTPException(status_code=401, detail="Missing device id")
    return device_id


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
