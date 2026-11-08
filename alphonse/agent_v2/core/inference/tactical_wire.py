"""Codex-compatible structured envelope for arbitrary native tool arguments."""
from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


QUESTION = _object({"question_id": {"type": "string"}, "type": {"type": "string", "enum": ["noul"]},
    "instructions": {"type": "string"}, "criteria": _object({"true": {"type": "string"}, "false": {"type": "string"}})})
ACTION = _object({"tool_id": {"type": "string"}, "arguments_json": {"type": "string"},
                  "acceptance_questions": {"type": "array", "items": QUESTION}})
ASSESSMENT = _object({"question_id": {"type": "string"},
    "answer": {"type": "string", "enum": ["supported", "not_supported", "uncertain"]},
    "evidence_refs": {"type": "array", "items": {"type": "string"}}})
SCHEMA = _object({"action": {"anyOf": [ACTION, {"type": "null"}]},
    "actions": {"type": "array", "items": ACTION},
    "browser_fill_sequence": {"type": "array", "items": ACTION},
    "uncertainty_assessment": {"anyOf": [ASSESSMENT, {"type": "null"}]}})
Draft202012Validator.check_schema(SCHEMA)
_VALIDATOR = Draft202012Validator(SCHEMA)

INSTRUCTIONS = ("For this provider return the structured envelope: set exactly one of action, actions, "
    "or browser_fill_sequence; use null for action when using an array, and empty arrays for unused sequences. "
    "Each action's arguments_json must be a complete JSON object encoded as a string with the exact native "
    "tool arguments. Set uncertainty_assessment to null unless requested. The executor validates all fields.")


def decode(value: Any) -> dict[str, Any]:
    if not _VALIDATOR.is_valid(value):
        raise ValueError("openai_codex_invalid_tactical_schema")
    variants = [value['action'] is not None, bool(value['actions']), bool(value['browser_fill_sequence'])]
    if sum(variants) != 1:
        raise ValueError("openai_codex_invalid_tactical_schema")

    def action(item: dict[str, Any]) -> dict[str, Any]:
        try:
            args = json.loads(item['arguments_json'])
        except (TypeError, ValueError):
            raise ValueError("openai_codex_invalid_json") from None
        if not isinstance(args, dict):
            raise ValueError("openai_codex_invalid_tactical_schema")
        return {"tool_id": item['tool_id'], "arguments": args,
                "acceptance_questions": item['acceptance_questions']}

    if variants[0]:
        result = action(value['action'])
    else:
        key = 'actions' if variants[1] else 'browser_fill_sequence'
        result = {key: [action(item) for item in value[key]]}
    if value['uncertainty_assessment'] is not None:
        result['uncertainty_assessment'] = value['uncertainty_assessment']
    return result
