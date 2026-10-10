"""Provider-independent execution telemetry with deterministic token estimates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


_SENSITIVE_KEY_PARTS = ("secret", "password", "authorization", "cookie", "api_key", "apikey")
_SENSITIVE_TOKEN_KEYS = ("token", "access_token", "refresh_token", "id_token", "api_token")


def approximate_tokens(value: str) -> int:
    """Estimate tokens consistently without assuming a provider tokenizer."""
    rendered = str(value or "")
    return 0 if not rendered else max(1, (len(rendered) + 3) // 4)


@dataclass(frozen=True)
class InferenceTelemetryEvent:
    event_type: str = "inference"
    purpose: str = ""
    task_id: str = ""
    project_id: str = ""
    provider: str = ""
    model: str = ""
    profile_id: str = ""
    input_chars: int = 0
    input_tokens_estimate: int = 0
    output_chars: int = 0
    output_tokens_estimate: int = 0
    tool_count: int = 0
    tool_schema_chars: int = 0
    tool_schema_tokens_estimate: int = 0
    duration_ms: int = 0
    status: str = "success"
    error: str = ""
    provider_usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolTelemetryEvent:
    event_type: str = "tool"
    tool_id: str = ""
    task_id: str = ""
    project_id: str = ""
    argument_chars: int = 0
    result_chars: int = 0
    duration_ms: int = 0
    status: str = "success"
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def json_chars(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def safe_telemetry_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Return bounded, JSON-safe telemetry metadata with likely secrets removed."""
    normalized_key = key.lower().replace("-", "_")
    if normalized_key in _SENSITIVE_TOKEN_KEYS or any(marker in normalized_key for marker in _SENSITIVE_KEY_PARTS):
        return "[redacted]"
    if depth >= 4:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 500 else f"{value[:499]}…"
    if isinstance(value, (list, tuple)):
        return [safe_telemetry_value(item, depth=depth + 1) for item in value[:25]]
    if isinstance(value, dict):
        return {
            str(item_key): safe_telemetry_value(item_value, key=str(item_key), depth=depth + 1)
            for item_key, item_value in list(value.items())[:50]
        }
    return safe_telemetry_value(str(value), key=key, depth=depth)
