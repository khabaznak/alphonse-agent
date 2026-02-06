from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / "alphonse" / "agent" / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def main() -> None:
    _load_env()
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    alphonse_id = os.getenv("ALPHONSE_ID", "alphonse-local")
    device_id = os.getenv("RELAY_DEVICE_ID", "device-smoke")
    if not supabase_url or not service_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    rest = f"{supabase_url.rstrip('/')}/rest/v1"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    channel_id = ensure_channel(rest, headers, alphonse_id, device_id)
    correlation_id = str(uuid.uuid4())
    command_id = str(uuid.uuid4())
    message = {
        "id": command_id,
        "channel_id": channel_id,
        "sender": "mobile",
        "type": "command",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "correlation_id": correlation_id,
        "device_id": device_id,
        "alphonse_id": alphonse_id,
        "payload": {"type": "request_status"},
        "schema_version": 1,
        "delivered_to_alphonse": False,
        "delivered_to_device": False,
    }
    resp = requests.post(f"{rest}/relay_messages", headers=headers, json=message, timeout=5)
    if resp.status_code >= 400:
        raise SystemExit(f"Insert command failed: {resp.status_code} {resp.text}")

    print("Inserted command, waiting for response...")
    for _ in range(20):
        params = {
            "alphonse_id": f"eq.{alphonse_id}",
            "sender": "eq.alphonse",
            "type": "eq.response",
            "correlation_id": f"eq.{correlation_id}",
            "order": "ts.desc",
            "limit": "1",
        }
        res = requests.get(f"{rest}/relay_messages", headers=headers, params=params, timeout=5)
        data = res.json() if res.status_code < 400 else []
        if data:
            print(json.dumps(data[0], indent=2))
            return
        time.sleep(0.5)
    raise SystemExit("No response received")


def ensure_channel(rest: str, headers: dict[str, str], alphonse_id: str, device_id: str) -> str:
    params = {
        "alphonse_id": f"eq.{alphonse_id}",
        "device_id": f"eq.{device_id}",
        "limit": "1",
    }
    resp = requests.get(f"{rest}/relay_channels", headers=headers, params=params, timeout=5)
    if resp.status_code >= 400:
        raise SystemExit(f"Fetch channel failed: {resp.status_code} {resp.text}")
    rows = resp.json()
    if rows:
        return rows[0]["id"]
    channel_id = str(uuid.uuid4())
    payload = {
        "id": channel_id,
        "alphonse_id": alphonse_id,
        "device_id": device_id,
    }
    created = requests.post(f"{rest}/relay_channels", headers=headers, json=payload, timeout=5)
    if created.status_code >= 400:
        raise SystemExit(f"Create channel failed: {created.status_code} {created.text}")
    return channel_id


if __name__ == "__main__":
    main()
