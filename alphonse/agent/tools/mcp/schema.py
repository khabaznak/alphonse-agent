from __future__ import annotations

from typing import Any


PROFILE_SCHEMA_VERSION = 1


def validate_profile_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be an object"]

    key = str(payload.get("key") or "").strip()
    description = str(payload.get("description") or "").strip()
    if not key:
        errors.append("key is required")
    if not description:
        errors.append("description is required")

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    supports_native = bool(metadata.get("native_tools"))
    operations = payload.get("operations")
    if not isinstance(operations, dict):
        errors.append("operations must be an object")
    elif not operations and not supports_native:
        errors.append("operations must be non-empty unless metadata.native_tools=true")
    else:
        for op_name, op_payload in operations.items():
            if not isinstance(op_payload, dict):
                errors.append(f"operation '{op_name}' must be an object")
                continue
            op_key = str(op_payload.get("key") or str(op_name) or "").strip()
            op_description = str(op_payload.get("description") or "").strip()
            command_template = str(op_payload.get("command_template") or "").strip()
            if not op_key:
                errors.append(f"operation '{op_name}' key is required")
            if not op_description:
                errors.append(f"operation '{op_name}' description is required")
            if not command_template:
                errors.append(f"operation '{op_name}' command_template is required")
            required_args = op_payload.get("required_args")
            if required_args is not None and (
                not isinstance(required_args, list)
                or any(not isinstance(item, str) or not str(item).strip() for item in required_args)
            ):
                errors.append(f"operation '{op_name}' required_args must be a list of non-empty strings")

    candidates = payload.get("binary_candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("binary_candidates must be a non-empty list")
    else:
        invalid_candidate = [
            item for item in candidates if not isinstance(item, str) or not str(item).strip()
        ]
        if invalid_candidate:
            errors.append("binary_candidates must contain non-empty strings")

    aliases = payload.get("aliases")
    if aliases is not None and (
        not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases)
    ):
        errors.append("aliases must be a list of strings")

    modes = payload.get("allowed_modes")
    if modes is not None and (
        not isinstance(modes, list) or any(not isinstance(item, str) or not str(item).strip() for item in modes)
    ):
        errors.append("allowed_modes must be a list of non-empty strings")

    fallback = payload.get("npx_package_fallback")
    if fallback is not None and (not isinstance(fallback, str) or not str(fallback).strip()):
        errors.append("npx_package_fallback must be a non-empty string when provided")

    version = payload.get("schema_version")
    if version is not None and int(version) != PROFILE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PROFILE_SCHEMA_VERSION}")
    return errors
