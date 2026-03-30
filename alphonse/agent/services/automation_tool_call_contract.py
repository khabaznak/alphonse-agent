from __future__ import annotations

from typing import Any

TOOL_CALL_CONTRACT_VERSION = 1

ERR_PAYLOAD_NOT_OBJECT = "tool_call_payload_not_object"
ERR_MISSING_TOOL_CALL = "tool_call_missing_container"
ERR_INVALID_TOOL_CALL = "tool_call_invalid_container"
ERR_INVALID_KIND = "tool_call_invalid_kind"
ERR_MISSING_TOOL_NAME = "tool_call_missing_tool_name"
ERR_INVALID_ARGS = "tool_call_invalid_args"
ERR_LEGACY_REJECTED = "tool_call_legacy_shape_rejected"


def build_canonical_tool_call_payload(*, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    rendered_tool = str(tool_name or "").strip()
    if not rendered_tool:
        raise ValueError(ERR_MISSING_TOOL_NAME)
    rendered_args = dict(args or {})
    return {
        "contract_version": TOOL_CALL_CONTRACT_VERSION,
        "tool_call": {
            "kind": "call_tool",
            "tool_name": rendered_tool,
            "args": rendered_args,
        },
    }


def is_canonical_tool_call(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    tool_call = payload.get("tool_call")
    if not isinstance(tool_call, dict):
        return False
    if str(tool_call.get("kind") or "").strip() != "call_tool":
        return False
    if not str(tool_call.get("tool_name") or "").strip():
        return False
    return isinstance(tool_call.get("args"), dict)


def to_canonical_tool_call(payload: Any, *, allow_legacy: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(ERR_PAYLOAD_NOT_OBJECT)
    if is_canonical_tool_call(payload):
        canonical = dict(payload)
        canonical.setdefault("contract_version", TOOL_CALL_CONTRACT_VERSION)
        return canonical

    if not allow_legacy:
        if "tool_call" not in payload:
            raise ValueError(ERR_MISSING_TOOL_CALL)
        raise ValueError(ERR_LEGACY_REJECTED)

    tool_name = _legacy_tool_name(payload)
    if not tool_name:
        if "tool_call" in payload and not isinstance(payload.get("tool_call"), dict):
            raise ValueError(ERR_INVALID_TOOL_CALL)
        raise ValueError(ERR_MISSING_TOOL_NAME)
    args = payload.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError(ERR_INVALID_ARGS)
    canonical = build_canonical_tool_call_payload(tool_name=tool_name, args=args)
    for key, value in payload.items():
        if key in {"tool_key", "tool_name", "tool", "args"}:
            continue
        canonical.setdefault(str(key), value)
    canonical["migration"] = {
        "kind": "tool_call_contract_v1",
        "source_shape": _legacy_source_shape(payload),
    }
    return canonical


def extract_canonical_call(payload: Any) -> tuple[str, dict[str, Any]]:
    canonical = to_canonical_tool_call(payload, allow_legacy=False)
    tool_call = canonical.get("tool_call")
    if not isinstance(tool_call, dict):
        raise ValueError(ERR_INVALID_TOOL_CALL)
    if str(tool_call.get("kind") or "").strip() != "call_tool":
        raise ValueError(ERR_INVALID_KIND)
    tool_name = str(tool_call.get("tool_name") or "").strip()
    if not tool_name:
        raise ValueError(ERR_MISSING_TOOL_NAME)
    args = tool_call.get("args")
    if not isinstance(args, dict):
        raise ValueError(ERR_INVALID_ARGS)
    return tool_name, dict(args)


def _legacy_tool_name(payload: dict[str, Any]) -> str:
    return str(payload.get("tool_key") or payload.get("tool_name") or payload.get("tool") or "").strip()


def _legacy_source_shape(payload: dict[str, Any]) -> str:
    if str(payload.get("tool_key") or "").strip():
        return "tool_key"
    if str(payload.get("tool_name") or "").strip():
        return "tool_name"
    if str(payload.get("tool") or "").strip():
        return "tool"
    return "unknown"
