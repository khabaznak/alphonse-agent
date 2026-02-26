from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from alphonse.agent.tools.mcp.loader import load_profile_payloads


@dataclass(frozen=True)
class McpOperationProfile:
    key: str
    description: str
    command_template: str
    required_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class McpServerProfile:
    key: str
    description: str
    binary_candidates: tuple[str, ...]
    operations: dict[str, McpOperationProfile]
    aliases: tuple[str, ...] = ()
    allowed_modes: tuple[str, ...] = ("ops", "dev")
    npx_package_fallback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class McpProfileRegistry:
    def __init__(self, profiles: list[McpServerProfile] | None = None) -> None:
        loaded = profiles if profiles is not None else _profiles_from_payloads(load_profile_payloads())
        self._by_key = {item.key: item for item in loaded}
        self._alias_to_key: dict[str, str] = {}
        for item in loaded:
            self._alias_to_key[item.key.lower()] = item.key
            for alias in item.aliases:
                self._alias_to_key[str(alias).strip().lower()] = item.key

    def get(self, profile_key: str) -> McpServerProfile | None:
        normalized = str(profile_key or "").strip().lower()
        if not normalized:
            return None
        canonical = self._alias_to_key.get(normalized)
        if not canonical:
            return None
        return self._by_key.get(canonical)

    def keys(self) -> list[str]:
        return sorted(self._by_key.keys())


def profile_keys() -> list[str]:
    return McpProfileRegistry().keys()


def _profiles_from_payloads(payloads: list[dict[str, Any]]) -> list[McpServerProfile]:
    profiles: list[McpServerProfile] = []
    for payload in payloads:
        operations_raw = payload.get("operations")
        operations: dict[str, McpOperationProfile] = {}
        if isinstance(operations_raw, dict):
            for op_name, op_payload in operations_raw.items():
                if not isinstance(op_payload, dict):
                    continue
                op_key = str(op_payload.get("key") or op_name).strip()
                if not op_key:
                    continue
                operations[op_key] = McpOperationProfile(
                    key=op_key,
                    description=str(op_payload.get("description") or "").strip(),
                    command_template=str(op_payload.get("command_template") or "").strip(),
                    required_args=tuple(str(item).strip() for item in op_payload.get("required_args") or [] if str(item).strip()),
                )
        metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {}
        supports_native = bool(metadata.get("native_tools"))
        key = str(payload.get("key") or "").strip()
        if not key:
            continue
        if not operations and not supports_native:
            continue
        profiles.append(
            McpServerProfile(
                key=key,
                description=str(payload.get("description") or "").strip(),
                binary_candidates=tuple(
                    str(item).strip() for item in payload.get("binary_candidates") or [] if str(item).strip()
                ),
                operations=operations,
                aliases=tuple(str(item).strip() for item in payload.get("aliases") or [] if str(item).strip()),
                allowed_modes=tuple(
                    str(item).strip() for item in payload.get("allowed_modes") or ("ops", "dev") if str(item).strip()
                ),
                npx_package_fallback=_normalize_optional_string(payload.get("npx_package_fallback")),
                metadata=metadata,
            )
        )
    return profiles


def _normalize_optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
