"""Advisory judgments about a draft relative to an active owner request.

No message is submitted and no authorization or effect follows from this result.
"""
from __future__ import annotations

import math

from alphonse.agent_v2.system_one import TypeSafeSystemOneClient

QUESTIONS = {
    'updates_active': {
        'type': 'noul',
        'instructions': 'Does `draft` ask the assistant to change or add context to the work described in `active_request` while it is underway?',
        'criteria': {
            'true': 'The sender intends the draft to modify or provide context for the active work.',
            'false': 'The sender intends a separate later request, or the draft does not indicate an update to the active work.',
        },
    },
    'separate_followup': {
        'type': 'noul',
        'instructions': 'Does `draft` express a separate request intended for after the work described in `active_request`?',
        'criteria': {
            'true': 'The sender intends a separate request to be handled after the active work.',
            'false': 'The sender intends an update to the active work, or the draft does not indicate a separate later request.',
        },
    },
}


def suggest(settings, *, active_request: str, draft: str, client=None) -> str | None:
    """Return steer/queue only for an unambiguous strong judgment; otherwise no suggestion."""
    if not settings.enabled or not settings.api_key or not settings.validated_at:
        return None
    if not isinstance(active_request, str) or not active_request.strip() or len(active_request.encode('utf-8')) > 8192:
        return None
    if not isinstance(draft, str) or not draft.strip() or len(draft.encode('utf-8')) > 4096:
        return None
    client = client or TypeSafeSystemOneClient(api_url=settings.api_url, api_key=settings.api_key,
                                                model=settings.model, timeout_seconds=3.0)
    try:
        response = client.evaluate(state={'active_request': active_request, 'draft': draft}, questions=QUESTIONS, retries=0)
        answers = response['answers']
        values = []
        for name in QUESTIONS:
            answer = answers[name]
            value = answer['noul'] if answer['type'] == 'noul' else None
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                return None
            values.append(value)
    except Exception:
        return None
    update, followup = values
    if update >= settings.yes_threshold and followup <= settings.no_threshold:
        return 'steer'
    if followup >= settings.yes_threshold and update <= settings.no_threshold:
        return 'queue'
    return None
