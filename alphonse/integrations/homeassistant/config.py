from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from alphonse.agent.nervous_system.tool_configs import get_active_tool_config


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay_sec: float = 0.4
    max_delay_sec: float = 3.0


@dataclass(frozen=True)
class WsReconnectConfig:
    open_timeout_sec: float = 10.0
    recv_timeout_sec: float = 30.0
    min_backoff_sec: float = 1.0
    max_backoff_sec: float = 30.0
    jitter_ratio: float = 0.2


@dataclass(frozen=True)
class DebounceConfig:
    enabled: bool = False
    window_ms: int = 500
    key_strategy: str = "entity_state"
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HomeAssistantConfig:
    base_url: str
    token: str
    request_timeout_sec: float = 10.0
    retry: RetryConfig = field(default_factory=RetryConfig)
    ws: WsReconnectConfig = field(default_factory=WsReconnectConfig)
    allowed_domains: frozenset[str] = field(default_factory=frozenset)
    allowed_entity_ids: frozenset[str] = field(default_factory=frozenset)
    debounce: DebounceConfig = field(default_factory=DebounceConfig)


class HomeAssistantConfigError(ValueError):
    pass


def load_homeassistant_config() -> HomeAssistantConfig | None:
    env_values = _env_config_values()
    store_values = _tool_config_values()
    values = {**env_values, **store_values}

    base_url = str(values.get("HA_BASE_URL") or "").strip().rstrip("/")
    token = str(values.get("HA_TOKEN") or "").strip()
    if not base_url or not token:
        return None
    if not base_url.startswith(("http://", "https://")):
        raise HomeAssistantConfigError("HA_BASE_URL must start with http:// or https://")

    return HomeAssistantConfig(
        base_url=base_url,
        token=token,
        request_timeout_sec=_as_float(values.get("HA_REQUEST_TIMEOUT_SEC"), 10.0, minimum=1.0),
        retry=RetryConfig(
            max_attempts=_as_int(values.get("HA_RETRY_MAX_ATTEMPTS"), 3, minimum=1),
            base_delay_sec=_as_float(values.get("HA_RETRY_BASE_DELAY_SEC"), 0.4, minimum=0.05),
            max_delay_sec=_as_float(values.get("HA_RETRY_MAX_DELAY_SEC"), 3.0, minimum=0.1),
        ),
        ws=WsReconnectConfig(
            open_timeout_sec=_as_float(values.get("HA_WS_OPEN_TIMEOUT_SEC"), 10.0, minimum=1.0),
            recv_timeout_sec=_as_float(values.get("HA_WS_RECV_TIMEOUT_SEC"), 30.0, minimum=1.0),
            min_backoff_sec=_as_float(values.get("HA_WS_RECONNECT_MIN_SEC"), 1.0, minimum=0.1),
            max_backoff_sec=_as_float(values.get("HA_WS_RECONNECT_MAX_SEC"), 30.0, minimum=0.2),
            jitter_ratio=_as_float(values.get("HA_WS_BACKOFF_JITTER_RATIO"), 0.2, minimum=0.0),
        ),
        allowed_domains=frozenset(_as_csv_set(values.get("HA_ALLOWED_DOMAINS"))),
        allowed_entity_ids=frozenset(_as_csv_set(values.get("HA_ALLOWED_ENTITY_IDS"))),
        debounce=DebounceConfig(
            enabled=_as_bool(values.get("HA_DEBOUNCE_ENABLED"), False),
            window_ms=_as_int(values.get("HA_DEBOUNCE_MS_DEFAULT"), 500, minimum=1),
            key_strategy=_as_debounce_strategy(values.get("HA_DEBOUNCE_KEY_STRATEGY")),
            attributes=tuple(sorted(_as_csv_set(values.get("HA_DEBOUNCE_ATTRIBUTES")))),
        ),
    )


def _env_config_values() -> dict[str, Any]:
    keys = {
        "HA_BASE_URL",
        "HA_TOKEN",
        "HA_REQUEST_TIMEOUT_SEC",
        "HA_RETRY_MAX_ATTEMPTS",
        "HA_RETRY_BASE_DELAY_SEC",
        "HA_RETRY_MAX_DELAY_SEC",
        "HA_WS_OPEN_TIMEOUT_SEC",
        "HA_WS_RECV_TIMEOUT_SEC",
        "HA_WS_RECONNECT_MIN_SEC",
        "HA_WS_RECONNECT_MAX_SEC",
        "HA_WS_BACKOFF_JITTER_RATIO",
        "HA_ALLOWED_DOMAINS",
        "HA_ALLOWED_ENTITY_IDS",
        "HA_DEBOUNCE_ENABLED",
        "HA_DEBOUNCE_MS_DEFAULT",
        "HA_DEBOUNCE_KEY_STRATEGY",
        "HA_DEBOUNCE_ATTRIBUTES",
    }
    return {key: os.getenv(key) for key in keys}


def _tool_config_values() -> dict[str, Any]:
    try:
        cfg = get_active_tool_config("homeassistant")
    except Exception:
        return {}
    if not cfg or not isinstance(cfg.get("config"), dict):
        return {}
    raw = cfg.get("config") or {}
    mapped: dict[str, Any] = {}
    for key in (
        "HA_BASE_URL",
        "HA_TOKEN",
        "HA_REQUEST_TIMEOUT_SEC",
        "HA_RETRY_MAX_ATTEMPTS",
        "HA_RETRY_BASE_DELAY_SEC",
        "HA_RETRY_MAX_DELAY_SEC",
        "HA_WS_OPEN_TIMEOUT_SEC",
        "HA_WS_RECV_TIMEOUT_SEC",
        "HA_WS_RECONNECT_MIN_SEC",
        "HA_WS_RECONNECT_MAX_SEC",
        "HA_WS_BACKOFF_JITTER_RATIO",
        "HA_ALLOWED_DOMAINS",
        "HA_ALLOWED_ENTITY_IDS",
        "HA_DEBOUNCE_ENABLED",
        "HA_DEBOUNCE_MS_DEFAULT",
        "HA_DEBOUNCE_KEY_STRATEGY",
        "HA_DEBOUNCE_ATTRIBUTES",
    ):
        if key in raw:
            mapped[key] = raw[key]
    return mapped


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(raw: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        return max(minimum, value)
    return value


def _as_float(raw: Any, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        return max(minimum, value)
    return value


def _as_csv_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items = raw
    else:
        items = str(raw).split(",")
    return {
        str(item).strip()
        for item in items
        if str(item).strip()
    }


def _as_debounce_strategy(raw: Any) -> str:
    value = str(raw or "entity_state").strip().lower()
    if value in {"entity", "entity_state", "entity_state_attributes"}:
        return value
    return "entity_state"
