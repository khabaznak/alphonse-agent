"""Bounded progress review; a judgment never authorizes continued work.

The caller must provide a trusted disclosure check for this exact recipient and
reauthorize each source. An absent, incomplete or oversized evidence set is
unknown, not proof of lack of progress. No truncation can turn it into complete
evidence. The model receives only data explicitly approved by that check.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable


def window_evidence_refs(current: list[str], baseline: set[str] | None) -> tuple[str, ...]:
    """Select this window only; an untrusted/missing baseline stays unknown."""
    if (baseline is None or not isinstance(baseline, set) or not isinstance(current, list)
            or not all(isinstance(ref, str) and ref for ref in current)
            or not baseline.issubset(current)):
        return ()
    return tuple(ref for ref in current if ref not in baseline)


def review_progress(*, request: str, observations: list[dict[str, Any]],
                    authorize_disclosure: Callable[[], bool] | None,
                    judge: Callable[..., Any] | None,
                    no_threshold: float, yes_threshold: float) -> str:
    """Return progress, no_progress, or unknown; never infer task completion."""
    if (not isinstance(request, str) or not request.strip() or not observations
            or not isinstance(observations, list)
            or not all(isinstance(item, dict) and set(item) == {'tool', 'arguments', 'observation'}
                       and isinstance(item['tool'], str) and isinstance(item['arguments'], dict)
                       and isinstance(item['observation'], dict) for item in observations)
            or not callable(authorize_disclosure) or not callable(judge)
            or type(no_threshold) not in (int, float) or type(yes_threshold) not in (int, float)
            or not 0 <= no_threshold < yes_threshold <= 1):
        return 'unknown'
    state = {'request': request, 'observations': observations}
    try:
        if len(json.dumps(state, ensure_ascii=False, allow_nan=False).encode('utf-8')) > 32768:
            return 'unknown'
        if authorize_disclosure() is not True:
            return 'unknown'
        probability = judge(state=state)
        if (type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            return 'unknown'
        # Revocation cannot retract data already sent, but it must prevent the
        # resulting assessment from being used after disclosure was revoked.
        if authorize_disclosure() is not True:
            return 'unknown'
    except Exception:
        return 'unknown'
    if probability >= yes_threshold:
        return 'progress'
    if probability <= no_threshold:
        return 'no_progress'
    return 'unknown'
