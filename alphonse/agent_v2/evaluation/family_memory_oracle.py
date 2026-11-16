"""Independent literal scoring for the frozen synthetic memory contrast.

This evaluator is used after a run; its expectations are never given to the
model. Exact source text in a prompt is not proof of model attention.
"""
from __future__ import annotations

from typing import Any


def score_heldout(case: dict[str, Any], answer: str, *,
                  full_source_text_visible: dict[str, bool],
                  explicitly_read: list[str]) -> dict[str, bool]:
    """Score answer and evidence using the predeclared synthetic fixture."""
    text = answer.casefold()
    checks = {
        'required_terms': all(term in text for term in case['required_terms']),
        'forbidden_terms': not any(term in text for term in case['forbidden_terms']),
    }
    answer_pass = bool(answer and all(checks.values()))
    required = case['required_sources']
    return {
        **checks,
        'answer_check_pass': answer_pass,
        'answer_and_full_source_text_check_pass': answer_pass and all(
            full_source_text_visible.get(name) is True for name in required),
        'strict_explicit_read_check_pass': answer_pass and all(
            name in explicitly_read for name in required),
    }
