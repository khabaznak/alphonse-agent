"""Live, synthetic-only Jev probe; not wired to the owner loop.

One persisted reservation per physical request. Do not add household data.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from .gym_budget import BudgetTransport

QUESTIONS = {
    'recap_conflicts': {'type': 'noul', 'instructions':
        'Does the assistant recap conflict with what the earlier owner statement says had been decided?'},
    'answer_acknowledges': {'type': 'noul', 'instructions':
        'Does the proposed answer acknowledge that the assistant recap overlooked the earlier owner statement?'},
    'answer_attributes': {'type': 'noul', 'instructions':
        'Does the proposed answer identify the assistant as the speaker of the recap?'},
    'answer_own_statement': {'type': 'noul', 'instructions':
        'Does the proposed answer refer to the recap as the assistant’s own earlier statement?'},
    'owner_changed': {'type': 'noul', 'instructions':
        'Does the later owner statement report changing the earlier decision?'},
}

# Explicitly synthetic, named fields; no real people, records or identifiers.
CASES = {
    'missed_recap': {
        'earlier_owner': 'We chose the striped umbrella for the imaginary picnic.',
        'later_owner': 'That earlier choice still stands.',
        'assistant_recap': 'We have not chosen an umbrella.',
        'proposed_answer': 'You chose the striped umbrella, but a later note says the choice is open.',
        'source_complete': True,
    },
    'acknowledged_recap': {
        'earlier_owner': 'We chose the striped umbrella for the imaginary picnic.',
        'later_owner': 'That earlier choice still stands.',
        'assistant_recap': 'We have not chosen an umbrella.',
        'proposed_answer': 'You chose the striped umbrella. My earlier recap overlooked that choice.',
        'source_complete': True,
    },
    'genuine_owner_change': {
        'earlier_owner': 'We chose a red kite for the imaginary festival.',
        'later_owner': 'I changed our choice to a yellow kite.',
        'assistant_recap': 'We chose a yellow kite.',
        'proposed_answer': 'You initially chose red, then changed it to yellow.',
        'source_complete': True,
    },
    'no_conflict': {
        'earlier_owner': 'The fictional club meets on Tuesday.',
        'later_owner': 'The Tuesday meeting is still planned.',
        'assistant_recap': 'The club meets on Tuesday.',
        'proposed_answer': 'The club meets Tuesday.',
        'source_complete': True,
    },
    'incomplete_earlier': {
        'earlier_owner': '[unavailable]',
        'later_owner': 'That earlier choice still stands.',
        'assistant_recap': 'We have not made a choice.',
        'proposed_answer': 'I cannot inspect the earlier choice.',
        'source_complete': False,
    },
    'spanish_contradiction': {
        'earlier_owner': 'Elegimos la mesa verde para el escenario ficticio.',
        'later_owner': 'La elección anterior sigue vigente.',
        'assistant_recap': 'Todavía no hemos elegido una mesa.',
        'proposed_answer': 'Sí, elegiste la mesa verde; mi respuesta anterior lo pasó por alto.',
        'source_complete': True,
    },
    'explicit_assistant_attribution': {
        'earlier_owner': 'The fictional train leaves at ten.',
        'later_owner': 'That plan has not changed.',
        'assistant_recap': 'We have not set a departure time.',
        'proposed_answer': 'The assistant previously said no time was set, overlooking your ten o’clock decision.',
        'source_complete': True,
    },
    'wrong_attribution': {
        'earlier_owner': 'The fictional train leaves at ten.',
        'later_owner': 'That plan has not changed.',
        'assistant_recap': 'We have not set a departure time.',
        'proposed_answer': 'You later said no time was set, although you originally chose ten.',
        'source_complete': True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path, help='new synthetic-only JSON journal')
    parser.add_argument('--settings-db', type=Path, default=Path.home() / '.alphonse' / 'alphonse-v2.sqlite3')
    args = parser.parse_args()
    if args.journal.exists():
        raise SystemExit('journal already exists')
    key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if not key:
        uri = args.settings_db.expanduser().resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute('SELECT api_key FROM v2_system_one_settings WHERE settings_id=1').fetchone()
            key = str(row[0] or '').strip() if row else ''
    if not key:
        raise SystemExit('No configured TypeSafe key')
    meter = BudgetTransport(args.journal, limit='0.03')
    for case_id, state in CASES.items():
        meter.local.case_id = case_id
        payload = {'model': 'jev-1.14.0', 'state': state, 'questions': QUESTIONS}
        try:
            meter('https://api.typesafe.ai/v1/systemone', key, payload, 10)
        except Exception as exc:
            print(f'{case_id}: request failed ({type(exc).__name__}); stopped')
            return
    print(f'Completed {len(meter.rows)} synthetic requests; conservative charge ${meter.charged}')


if __name__ == '__main__':
    main()
