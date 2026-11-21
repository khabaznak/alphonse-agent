"""Ask for a reminder without a time through the synthetic owner worker.

Examples test general behavior; they are not production vocabulary rules.
The independent observer checks effects, not the meaning of the reply.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from alphonse.agent_v2.evaluation.owner_dojo_shopping import run
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders


def scenario(language):
    requests = {'en': 'Please remind me to call the dentist.',
                'es': 'Recuérdame llamar al dentista.'}
    if language not in requests:
        raise ValueError('dojo_language_invalid')

    def readback_runtime(path, owner, started_at, runtime):
        return {'open_reminders': OwnerDesktopReminders(str(path)).list_open(owner),
                'scheduled_tasks': [r.to_dict() for r in runtime.owner_schedule_store.list_tasks(owner_user_id=owner)],
                'started_at': started_at.isoformat()}

    def grade(observation):
        state = observation.get('runtime')
        if not isinstance(state, dict):
            return {'effect_passed': None, 'clarification_semantics': 'unassessed'}
        no_effect = state['open_reminders'] == [] and state['scheduled_tasks'] == []
        return {'effect_passed': no_effect, 'no_scheduled_effect': no_effect,
                'clarification_semantics': 'unassessed'}

    return SimpleNamespace(CASE_ID='ambiguous-reminder-' + language, record_decision_text=True,
        GOAL=requests[language], seed=lambda _shell: None,
        readback=lambda _shell: {'status': 'observed'},
        readback_runtime=readback_runtime, grade=grade)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['en', 'es'])
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--synthetic-clarification-review', action='store_true',
                        help='stub the effect-claim judgment to allow; not a Jev test')
    args = parser.parse_args()
    case = scenario(args.language)
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(row):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': case.CASE_ID, 'goal': case.GOAL,
              'model': args.model, 'inference': 'openai_codex', 'image_id': args.image_id,
              'synthetic_clarification_review': args.synthetic_clarification_review,
              'source_tree_sha256': digest.hexdigest(), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'synthetic owner worker; disposable container Bash'})
        fd = os.open(args.journal.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                         model=args.model, timeout_seconds=120, scenario=case, full_registry=True,
                         synthetic_clarification_review=args.synthetic_clarification_review)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'task_passed': None,
                  'error_type': type(exc).__name__})
            raise
        result['limitations'].extend([
            'No-effect readback does not establish that the response asks a useful question.',
            'Two example languages do not establish multilingual coverage.',
        ])
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
