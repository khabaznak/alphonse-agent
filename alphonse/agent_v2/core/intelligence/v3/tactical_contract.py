"""Tactical response wire contract; authority is enforced separately at dispatch."""
from __future__ import annotations

from typing import Any
from jsonschema import Draft202012Validator

_QUESTION = {
    "type": "object", "required": ["question_id", "type", "instructions", "criteria"],
    "properties": {
        "question_id": {"type": "string", "minLength": 1}, "type": {"const": "noul"},
        "instructions": {"type": "string", "minLength": 1},
        "criteria": {"type": "object", "required": ["true", "false"],
            "properties": {"true": {"type": "string", "minLength": 1},
                           "false": {"type": "string", "minLength": 1}}, "additionalProperties": False},
    }, "additionalProperties": False,
}
_ACTION_PROPERTIES = {
    "tool_id": {"type": "string", "minLength": 1}, "arguments": {"type": "object"},
    "action_id": {"type": "string"},
    "acceptance_questions": {"type": "array", "minItems": 1, "items": _QUESTION},
}
_ACTION_REQUIRED = ["tool_id", "arguments", "acceptance_questions"]
_ACTION = {"type": "object", "properties": _ACTION_PROPERTIES, "required": _ACTION_REQUIRED,
           "additionalProperties": False}

TACTICAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {**_ACTION_PROPERTIES,
        "actions": {"type": "array", "minItems": 2, "maxItems": 3, "items": _ACTION},
        "browser_fill_sequence": {"type": "array", "minItems": 3, "maxItems": 3, "items": _ACTION},
        "uncertainty_assessment": {"type": "object", "required": ["question_id", "answer", "evidence_refs"],
            "properties": {"question_id": {"type": "string"},
                "answer": {"enum": ["supported", "not_supported", "uncertain"]},
                "evidence_refs": {"type": "array", "items": {"type": "string"}}},
            "additionalProperties": False}},
    "additionalProperties": False,
    "oneOf": [{"required": _ACTION_REQUIRED}, {"required": ["actions"]},
              {"required": ["browser_fill_sequence"]}],
}
Draft202012Validator.check_schema(TACTICAL_SCHEMA)
_VALIDATOR = Draft202012Validator(TACTICAL_SCHEMA)


def valid_tactical_response(value: Any) -> bool:
    """Do not expose schema validation messages containing model output."""
    return _VALIDATOR.is_valid(value)
