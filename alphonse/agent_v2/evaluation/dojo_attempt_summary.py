"""Summarize recorded Dojo observations without turning unknowns into passes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(events):
    """Summarize event facts; answer content and model semantics need review."""
    starts = [event for event in events if event.get('event') == 'started']
    finishes = [event for event in events if event.get('event') == 'finished']
    decisions = [event for event in events if event.get('event') == 'model_decision']
    bash_returns = [event for event in events if event.get('event') == 'bash_finished']
    last = finishes[-1] if finishes else {}
    usage = Counter()
    for decision in decisions:
        for name in ('input_tokens', 'output_tokens'):
            value = (decision.get('usage') or {}).get(name)
            if type(value) is int and value >= 0:
                usage[name] += value
    return {
        'case': (starts[0].get('case') if starts else None) or last.get('case'),
        'journal_complete': len(finishes) == 1,
        'owner_status': last.get('status'),
        'failure_code': last.get('failure_code'),
        'model_decisions': len(decisions),
        'model_usage_observed': dict(usage),
        'tool_choices': dict(Counter(d['tool_id'] for d in decisions if isinstance(d.get('tool_id'), str))),
        'bash_processes_started': sum(e.get('event') == 'bash_started' for e in events),
        'bash_results': dict(Counter(str(e.get('status', 'unknown')) for e in bash_returns)),
        'provider_failures': [e.get('category', 'unknown') for e in events if e.get('event') == 'model_failed'],
        'effect_passed': last.get('effect_passed'),
        'fixture_integrity': last.get('fixture_integrity'),
        'response_semantics': last.get('response_semantics', 'unassessed'),
        'recovery_semantics': last.get('recovery_semantics', 'unassessed'),
        'client_acknowledged': last.get('client_acknowledged'),
        'cleanup_verified': last.get('cleanup_verified'),
        'elapsed_seconds': last.get('elapsed_seconds'),
        'limitations': last.get('limitations', []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    args = parser.parse_args()
    with args.journal.open(encoding='utf-8') as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    print(json.dumps(summarize(events), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
