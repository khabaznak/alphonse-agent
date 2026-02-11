from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepValidationErrorType(str, Enum):
    INVALID_JSON_STEP = "INVALID_JSON_STEP"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    INCOMPLETE_PARAMETER = "INCOMPLETE_PARAMETER"
    INVALID_PARAMETER_VALUE = "INVALID_PARAMETER_VALUE"
    DISALLOWED_INTERNAL_QUESTION = "DISALLOWED_INTERNAL_QUESTION"
    UNBOUND_VARIABLE = "UNBOUND_VARIABLE"


@dataclass(frozen=True)
class StepValidationIssue:
    error_type: StepValidationErrorType
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepValidationResult:
    is_valid: bool
    issue: StepValidationIssue | None = None
    error_history: list[str] = field(default_factory=list)


_PLACEHOLDER_PATTERNS = (
    r"<[^>]+>",
    r"\byour\s+\w+\s+name\b",
    r"\bplaceholder\b",
    r"\[unknown\]",
)

_TOOL_CALL_PATTERN = re.compile(r"\b\w+\([^)]*\)")
_UNBOUND_VAR_PATTERN = re.compile(r"\$[A-Za-z_]\w*")


def validate_step(
    step: dict[str, Any],
    catalog: dict[str, Any] | None,
    *,
    error_history: list[str] | None = None,
) -> StepValidationResult:
    history = list(error_history or [])
    if not isinstance(step, dict):
        return _invalid(
            StepValidationErrorType.INVALID_JSON_STEP,
            "Step must be an object.",
            history=history,
        )
    tool = str(step.get("tool") or "").strip()
    if not tool:
        return _invalid(
            StepValidationErrorType.INVALID_JSON_STEP,
            "Step is missing a non-empty tool name.",
            history=history,
        )
    parameters = step.get("parameters")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        return _invalid(
            StepValidationErrorType.INVALID_JSON_STEP,
            "Step parameters must be an object.",
            details={"tool": tool},
            history=history,
        )

    tool_specs = _tool_spec_map(catalog)
    spec = tool_specs.get(tool)
    if spec is None:
        return _invalid(
            StepValidationErrorType.TOOL_NOT_FOUND,
            f"Tool '{tool}' is not available in current catalog.",
            details={"tool": tool, "available_tools": sorted(tool_specs.keys())},
            history=history,
        )

    required_params = _required_parameters(spec)
    missing = [
        name
        for name in required_params
        if name not in parameters or _is_empty(parameters.get(name))
    ]
    if missing:
        return _invalid(
            StepValidationErrorType.INCOMPLETE_PARAMETER,
            f"Missing required parameter(s): {', '.join(missing)}.",
            details={"tool": tool, "missing_parameters": missing},
            history=history,
        )

    for key, value in parameters.items():
        if _is_empty(value):
            return _invalid(
                StepValidationErrorType.INCOMPLETE_PARAMETER,
                f"Parameter '{key}' is empty.",
                details={"tool": tool, "parameter": key},
                history=history,
            )
        if isinstance(value, str):
            if _contains_placeholder(value):
                return _invalid(
                    StepValidationErrorType.INVALID_PARAMETER_VALUE,
                    f"Parameter '{key}' contains placeholder text.",
                    details={"tool": tool, "parameter": key},
                    history=history,
                )
            if _UNBOUND_VAR_PATTERN.search(value):
                return _invalid(
                    StepValidationErrorType.UNBOUND_VARIABLE,
                    f"Parameter '{key}' contains an unbound variable.",
                    details={"tool": tool, "parameter": key, "value": value},
                    history=history,
                )
            if _TOOL_CALL_PATTERN.search(value) and key.lower() not in {"taskcall"}:
                return _invalid(
                    StepValidationErrorType.INVALID_PARAMETER_VALUE,
                    f"Parameter '{key}' should not contain a tool-call expression.",
                    details={"tool": tool, "parameter": key, "value": value},
                    history=history,
                )

    if tool == "askQuestion":
        question = str(parameters.get("question") or "").strip()
        if not question:
            return _invalid(
                StepValidationErrorType.INCOMPLETE_PARAMETER,
                "askQuestion requires a non-empty 'question' parameter.",
                details={"tool": tool},
                history=history,
            )
        if is_internal_tool_question(question):
            return _invalid(
                StepValidationErrorType.DISALLOWED_INTERNAL_QUESTION,
                "askQuestion cannot ask users to choose internal tool/function names.",
                details={"tool": tool, "question": question},
                history=history,
            )

    return StepValidationResult(is_valid=True, issue=None, error_history=history)


def is_internal_tool_question(question: str) -> bool:
    normalized = str(question or "").lower().strip()
    if not normalized:
        return True
    forbidden = (
        "tool",
        "function",
        "api",
        "endpoint",
        "method",
        "setreminder",
        "reminderset",
        "notification tool",
    )
    if any(token in normalized for token in forbidden):
        return True
    return bool(
        re.search(r"\b(confirm|choose|pick|specify)\b.{0,24}\b(tool|function|api)\b", normalized)
    )


def _tool_spec_map(catalog: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(catalog, dict):
        return {}
    raw = catalog.get("tools")
    if not isinstance(raw, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or "").strip()
        if not name:
            continue
        mapped[name] = item
    return mapped


def _required_parameters(spec: dict[str, Any]) -> list[str]:
    explicit = spec.get("required_parameters")
    if isinstance(explicit, list):
        return [str(item).strip() for item in explicit if str(item).strip()]
    params = spec.get("input_parameters")
    if not isinstance(params, list):
        return []
    required: list[str] = []
    for item in params:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("required", False)):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            required.append(name)
    return required


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _contains_placeholder(value: str) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in _PLACEHOLDER_PATTERNS)


def _invalid(
    error_type: StepValidationErrorType,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    history: list[str],
) -> StepValidationResult:
    next_history = list(history)
    next_history.append(error_type.value)
    return StepValidationResult(
        is_valid=False,
        issue=StepValidationIssue(
            error_type=error_type,
            message=message,
            details=details or {},
        ),
        error_history=next_history,
    )
