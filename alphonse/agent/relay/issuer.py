from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any

import requests


def mint_relay_token(device_id: str, device_name: str | None = None) -> dict[str, Any] | None:
    secret = os.getenv("RELAY_JWT_SECRET")
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    alphonse_id = os.getenv("ALPHONSE_ID")
    if not (secret and supabase_url and service_key and alphonse_id):
        return None

    channel_id = _ensure_channel(
        supabase_url=supabase_url,
        service_key=service_key,
        alphonse_id=alphonse_id,
        device_id=device_id,
    )
    if not channel_id:
        return None

    now = int(time.time())
    exp = now + int(os.getenv("RELAY_TOKEN_TTL_SECS", "604800"))
    payload = {
        "channel_id": channel_id,
        "device_id": device_id,
        "device_name": device_name,
        "alphonse_id": alphonse_id,
        "iat": now,
        "exp": exp,
    }
    token = _encode_jwt(payload, secret)
    return {"relay_token": token, "channel_id": channel_id, "expires_at": exp}


def _ensure_channel(
    *,
    supabase_url: str,
    service_key: str,
    alphonse_id: str,
    device_id: str,
) -> str | None:
    rest = f"{supabase_url.rstrip('/')}/rest/v1"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    params = {
        "alphonse_id": f"eq.{alphonse_id}",
        "device_id": f"eq.{device_id}",
        "limit": "1",
    }
    res = requests.get(f"{rest}/relay_channels", headers=headers, params=params, timeout=5)
    if res.status_code < 400:
        rows = res.json()
        if rows:
            return rows[0]["id"]
    channel_id = str(uuid.uuid4())
    payload = {"id": channel_id, "alphonse_id": alphonse_id, "device_id": device_id}
    created = requests.post(f"{rest}/relay_channels", headers=headers, json=payload, timeout=5)
    if created.status_code >= 400:
        return None
    return channel_id


def _encode_jwt(payload: dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    head_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{head_b64}.{payload_b64}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url(sig)
    return f"{head_b64}.{payload_b64}.{sig_b64}"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")
