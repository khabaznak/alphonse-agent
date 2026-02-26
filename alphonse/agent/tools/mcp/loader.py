from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from alphonse.agent.tools.mcp.schema import validate_profile_payload


def default_profiles_dir() -> Path:
    override = str(os.getenv("ALPHONSE_MCP_PROFILES_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parent / "profiles").resolve()


def load_profile_payloads() -> list[dict[str, Any]]:
    root = default_profiles_dir()
    if not root.exists() or not root.is_dir():
        return []
    payloads: list[dict[str, Any]] = []
    for candidate in sorted(root.glob("*.json")):
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        errors = validate_profile_payload(raw)
        if errors:
            continue
        payloads.append(dict(raw))
    return payloads

