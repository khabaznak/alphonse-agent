from __future__ import annotations

from typing import Any


def validate_plan(schema: dict, data: dict) -> list[str]:
    errors: list[str] = []
    _validate_schema(schema, data, path="$", errors=errors)
    return errors


def _validate_schema(schema: dict, value: Any, *, path: str, errors: list[str]) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if not any(_matches_type(t, value) for t in schema_type):
            errors.append(f"{path}: type mismatch")
            return
    elif schema_type and not _matches_type(schema_type, value):
        errors.append(f"{path}: type mismatch")
        return

    if schema_type == "object" and isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing {key}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                _validate_schema(subschema, value[key], path=f"{path}.{key}", errors=errors)
        const = schema.get("const")
        if const is not None and value != const:
            errors.append(f"{path}: expected {const}")
    if schema_type == "array" and isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for idx, item in enumerate(value):
                _validate_schema(items, item, path=f"{path}[{idx}]", errors=errors)
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']}")


def _matches_type(schema_type: str, value: Any) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True
