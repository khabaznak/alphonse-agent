from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket


class LanConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, device_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[device_id] = websocket

    def disconnect(self, device_id: str) -> None:
        self._connections.pop(device_id, None)

    async def send(self, device_id: str, message: dict[str, Any]) -> None:
        websocket = self._connections.get(device_id)
        if not websocket:
            return
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(device_id)

    async def send_ack(self, device_id: str, action: str) -> None:
        await self.send(
            device_id,
            {
                "type": "ack",
                "timestamp": _utcnow_iso(),
                "payload": {"for": action},
            },
        )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
