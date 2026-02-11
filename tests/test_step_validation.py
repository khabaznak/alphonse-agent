from __future__ import annotations

from alphonse.agent.cognition.step_validation import (
    StepValidationErrorType,
    is_internal_tool_question,
    validate_step,
)


def _catalog() -> dict:
    return {
        "tools": [
            {
                "tool": "askQuestion",
                "required_parameters": ["question"],
                "input_parameters": [{"name": "question", "type": "string", "required": True}],
            },
            {
                "tool": "core.location.current",
                "required_parameters": [],
                "input_parameters": [],
            },
        ]
    }


def test_validate_step_rejects_unknown_tool() -> None:
    result = validate_step({"tool": "setTimer", "parameters": {"time": "1min"}}, _catalog())
    assert result.is_valid is False
    assert result.issue is not None
    assert result.issue.error_type == StepValidationErrorType.TOOL_NOT_FOUND


def test_validate_step_rejects_placeholder_value() -> None:
    result = validate_step(
        {"tool": "core.location.current", "parameters": {"label": "<Your Contact Name>"}},
        _catalog(),
    )
    assert result.is_valid is False
    assert result.issue is not None
    assert result.issue.error_type == StepValidationErrorType.INVALID_PARAMETER_VALUE


def test_validate_step_rejects_internal_tool_question() -> None:
    result = validate_step(
        {
            "tool": "askQuestion",
            "parameters": {"question": "Please confirm which tool to use."},
        },
        _catalog(),
    )
    assert result.is_valid is False
    assert result.issue is not None
    assert result.issue.error_type == StepValidationErrorType.DISALLOWED_INTERNAL_QUESTION


def test_validate_step_rejects_unbound_variable() -> None:
    result = validate_step(
        {"tool": "core.location.current", "parameters": {"label": "$remindTime"}},
        _catalog(),
    )
    assert result.is_valid is False
    assert result.issue is not None
    assert result.issue.error_type == StepValidationErrorType.UNBOUND_VARIABLE


def test_internal_tool_question_helper() -> None:
    assert is_internal_tool_question("Choose the tool please.") is True
    assert is_internal_tool_question("When should I remind you?") is False
