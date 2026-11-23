"""Reviewable evidence for synthetic Dojo runs, not a production guard.

Model decisions are intentions; fixture-service requests and completed Bash
observations are independent signals. Neither establishes a complete account
of provider-side activity. The report intentionally leaves answer semantics to
review rather than using missing records as proof that nothing happened.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


def summarize_synthetic_trace(events: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    """Summarize an owner Dojo run without conflating selection and dispatch.

    Only call for a scenario containing exclusively disposable synthetic data.
    Do not pass an inferred complete history to the semantic action-claim probe.
    """
    if not isinstance(events, list) or not all(isinstance(x, dict) for x in events) or not isinstance(result, dict):
        raise ValueError('dojo_trace_invalid')
    selections = []
    model_failures = []
    bash_started = []
    bash_finished = []
    provider_events = []
    for event in events:
        if event.get('event') == 'model_decision' and event.get('kind') == 'tool':
            selections.append({'tool_id': event.get('tool_id'), 'arguments': deepcopy(event.get('arguments'))})
        elif event.get('event') == 'model_failed':
            model_failures.append(event.get('category'))
        elif event.get('event') == 'bash_started':
            bash_started.append({'command': event.get('command')})
        elif event.get('event') == 'bash_finished':
            bash_finished.append({k: deepcopy(v) for k, v in event.items() if k != 'event'})
        elif event.get('event') == 'provider_event_type':
            provider_events.append({'event_type': event.get('event_type'), 'item_type': event.get('item_type')})
    readback = result.get('observed_readback')
    observed = readback.get('requests') if isinstance(readback, dict) else None
    request_records_available = (isinstance(readback, dict) and readback.get('status') == 'observed'
                                 and isinstance(observed, list)
                                 and all(isinstance(item, dict) for item in observed))
    return {
        'answer': result.get('answer') if isinstance(result.get('answer'), str) else None,
        'native_tool_selections': selections,
        'fixture_service_requests': deepcopy(observed) if request_records_available else None,
        'bash_starts': bash_started,
        'bash_finishes': bash_finished,
        'model_failures': model_failures,
        'provider_event_types': provider_events,
        'fixture_service_observation': 'available' if request_records_available else 'unavailable',
        # Codex inference is a separate process, and this run does not attest
        # all its actions. Even a scripted decision is not proof of dispatch.
        'provider_action_history': 'incomplete',
        'claim_assessment': 'manual_review_required',
        'limitations': [
            'A model selection is not proof a native tool was dispatched.',
            'An absent fixture request is not proof that no other route or provider-side action occurred.',
            'This evidence does not assess whether claims in the answer are true.',
        ],
    }


def summarize_six_facilities_journal(path: Path) -> dict[str, Any]:
    """Read a completed, explicitly synthetic six-facilities journal."""
    try:
        events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError('dojo_trace_journal_unavailable') from None
    if (not events or not all(isinstance(row, dict) for row in events)
            or events[0].get('event') != 'started'
            or events[0].get('case') != 'six-facility-current-reports-v2'
            or events[0].get('scope') != 'synthetic owner and search service; isolated container Bash'
            or events[-1].get('event') != 'finished'
            or events[-1].get('case') != events[0]['case']
            or sum(row.get('event') == 'finished' for row in events) != 1):
        raise ValueError('dojo_trace_journal_unavailable')
    return summarize_synthetic_trace(events[:-1], events[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path, help='completed synthetic six-facilities JSONL journal')
    args = parser.parse_args()
    print(json.dumps(summarize_six_facilities_journal(args.journal), ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
