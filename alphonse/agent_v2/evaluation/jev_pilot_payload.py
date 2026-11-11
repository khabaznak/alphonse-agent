"""Canonical request body bytes shared by the Jev pilot sender and meter."""
from __future__ import annotations

import json
from typing import Any


def serialize_payload(payload: dict[str, Any]) -> bytes:
    """Use deterministic UTF-8 JSON; reject non-finite values before dispatch."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
