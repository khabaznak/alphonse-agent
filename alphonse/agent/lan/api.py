from __future__ import annotations

from datetime import datetime, timezone, timedelta
import logging
import os
import secrets
import uuid
from typing import Any

from fastapi import APIRouter, Body, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from alphonse.agent.lan.qr import render_svg_qr
from alphonse.agent.relay.issuer import mint_relay_token
from alphonse.agent.lan.pairing_store import (
    create_pairing_request,
    get_pairing_request,
    mark_expired,
    append_audit,
)
from alphonse.agent.lan.pairing_notify import notify_pairing_request
from alphonse.brain.orchestrator import handle_event as handle_habit_event
from alphonse.agent.lan.store import set_device_token
from alphonse.agent.lan.store import (
    DEFAULT_ALLOWED_SCOPES,
    arm_device,
    consume_pairing_code,
    disarm_device,
    generate_pairing_code,
    get_latest_last_seen,
    get_paired_device,
    get_latest_paired_device,
    is_paired_device,
    list_paired_devices,
    register_device,
    update_device_last_seen,
    update_device_status,
)
from alphonse.agent.lan.ws import LanConnectionManager

logger = logging.getLogger(__name__)


router = APIRouter(tags=["lan"])
_connections = LanConnectionManager()


class PairRequest(BaseModel):
    pair_code: str
    device_id: str
    device_name: str | None = None


class PairingCodeResponse(BaseModel):
    pair_code: str
    expires_at: str


class PairingCodeQrResponse(BaseModel):
    pair_code: str
    expires_at: str
    qr_svg: str


class ArmRequest(BaseModel):
    device_id: str | None = None


class RelayTokenRequest(BaseModel):
    device_id: str
    device_name: str | None = None


class PairStartRequest(BaseModel):
    device_name: str
    device_platform: str | None = None


class PairCompleteRequest(BaseModel):
    pairing_id: str
    response_to_challenge: str | None = None


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


@router.post("/pairing-codes/qr", response_model=PairingCodeQrResponse)
def create_pairing_code_with_qr() -> PairingCodeQrResponse:
    ttl_minutes = _pairing_ttl_minutes()
    code = generate_pairing_code(ttl_minutes=ttl_minutes)
    qr_svg = render_svg_qr(code.code)
    return PairingCodeQrResponse(
        pair_code=code.code,
        expires_at=code.expires_at.isoformat(),
        qr_svg=qr_svg,
    )


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
    relay = mint_relay_token(device.device_id, device.device_name)
    return {
        "status": "paired",
        "device_id": device.device_id,
        "device_name": device.device_name,
        "paired_at": device.paired_at.isoformat(),
        "allowed_scopes": device.allowed_scopes,
        "relay": relay,
    }


@router.get("/status")
def lan_status(x_alphonse_device_id: str | None = Header(default=None)) -> dict[str, Any]:
    last_seen = get_latest_last_seen()
    device = None
    if x_alphonse_device_id:
        device = get_paired_device(x_alphonse_device_id)
    latest = get_latest_paired_device()
    return {
        "status": "ok",
        "alphonse_online": True,
        "version": os.getenv("ALPHONSE_VERSION", "dev"),
        "last_seen_device": last_seen.isoformat() if last_seen else None,
        "device": _device_payload(device) if device else None,
        "latest_device": _device_payload(latest) if latest else None,
    }


@router.get("/devices")
def list_devices() -> dict[str, Any]:
    devices = list_paired_devices(limit=100)
    return {
        "devices": [_device_payload(device) for device in devices]
    }


@router.post("/arm")
def arm(payload: ArmRequest) -> dict[str, Any]:
    device = _resolve_device(payload.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    arm_device(device.device_id, armed_by="telegram")
    device = get_paired_device(device.device_id)
    return {"status": "armed", "device": _device_payload(device)}


@router.post("/disarm")
def disarm(payload: ArmRequest) -> dict[str, Any]:
    device = _resolve_device(payload.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    disarm_device(device.device_id)
    device = get_paired_device(device.device_id)
    return {"status": "disarmed", "device": _device_payload(device)}


@router.post("/relay-token")
def relay_token(payload: RelayTokenRequest) -> dict[str, Any]:
    device = get_paired_device(payload.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    relay = mint_relay_token(payload.device_id, payload.device_name or device.device_name)
    if not relay:
        raise HTTPException(status_code=503, detail="Relay not configured")
    return relay


@router.post("/pair/start")
def pair_start(payload: PairStartRequest) -> dict[str, Any]:
    ttl = _pairing_ttl_minutes()
    request, otp = create_pairing_request(payload.device_name, ttl)
    try:
        handle_habit_event(
            "pairing.requested",
            {"severity": "critical", "requires_ack": True, "ttl_sec": ttl * 60},
            {
                "pairing_id": request.pairing_id,
                "device_name": request.device_name,
                "otp": otp,
                "expires_at": request.expires_at.isoformat(),
            },
        )
    except Exception as exc:
        logger.warning("Habit pipeline failed for pairing.requested: %s", exc)
        notify_pairing_request(
            request.pairing_id,
            request.device_name,
            otp,
            request.expires_at.isoformat(),
        )
    append_audit(
        "pairing.requested",
        request.pairing_id,
        {"device_name": payload.device_name, "device_platform": payload.device_platform},
    )
    return {
        "pairing_id": request.pairing_id,
        "challenge": request.challenge,
        "expires_at": request.expires_at.isoformat(),
    }


@router.post("/pair/complete")
def pair_complete(payload: PairCompleteRequest) -> dict[str, Any]:
    request = get_pairing_request(payload.pairing_id)
    if not request:
        raise HTTPException(status_code=404, detail="Pairing not found")
    if request.expires_at <= _utcnow():
        mark_expired(request.pairing_id)
        raise HTTPException(status_code=403, detail="Pairing expired")
    if request.status != "approved":
        raise HTTPException(status_code=403, detail=f"Pairing {request.status}")
    if payload.response_to_challenge and payload.response_to_challenge != request.challenge:
        raise HTTPException(status_code=403, detail="Invalid challenge response")

    device_id = str(uuid.uuid4())
    device = register_device(device_id=device_id, device_name=request.device_name)
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    token_ttl_days = int(os.getenv("PAIRING_TOKEN_TTL_DAYS", "30"))
    expires_at = _utcnow() + timedelta(days=token_ttl_days)
    set_device_token(device.device_id, token_hash, expires_at)
    append_audit(
        "pairing.completed",
        request.pairing_id,
        {"device_id": device.device_id, "device_name": device.device_name},
    )
    return {
        "relay_base_url": os.getenv("RELAY_BASE_URL") or os.getenv("SUPABASE_URL") or "",
        "device_id": device.device_id,
        "device_token": token,
        "expires_at": expires_at.isoformat(),
        "home_id": os.getenv("HOME_ID"),
        "capabilities": ["relay", "lan"],
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
    raw = os.getenv("PAIRING_TTL_MINUTES") or os.getenv("ALPHONSE_PAIRING_TTL_MINUTES", "15")
    try:
        ttl = int(raw)
    except ValueError:
        return 15
    return max(ttl, 1)


def _require_device_id(device_id: str | None) -> str:
    if not device_id:
        raise HTTPException(status_code=401, detail="Missing device id")
    return device_id


def _resolve_device(device_id: str | None):
    if device_id:
        return get_paired_device(device_id)
    return get_latest_paired_device()


def _device_payload(device) -> dict[str, Any]:
    if not device:
        return {}
    return {
        "device_id": device.device_id,
        "device_name": device.device_name,
        "paired_at": device.paired_at.isoformat(),
        "allowed_scopes": device.allowed_scopes,
        "armed": device.armed,
        "armed_at": device.armed_at.isoformat() if device.armed_at else None,
        "armed_by": device.armed_by,
        "armed_until": device.armed_until.isoformat() if device.armed_until else None,
        "last_seen_at": device.last_seen_at.isoformat() if device.last_seen_at else None,
        "last_status": device.last_status,
        "last_status_at": device.last_status_at.isoformat() if device.last_status_at else None,
    }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode()).hexdigest()
