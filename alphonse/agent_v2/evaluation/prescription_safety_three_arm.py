"""Scripted processor plumbing for public synthetic prescription safety cases.

No live model, clinical conclusion, or real communication is involved. Three
processors receive the same preselected answer and read the same fixture data.
The scenario's evaluator independently checks observed source values and exact
dose/time output; the answer is not generated autonomously.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .prescription_safety_development import CASES, PrescriptionFixture, evaluate, task_request
from .prescription_v3_adapter import PrescriptionV3Adapter
from .prescription_candidate_adapter import PrescriptionCandidateAdapter

_CASE_ID = 'development-prescription-morning-day-three'
_INDEX = 'household/index.md'
_PRESCRIPTION = 'household/prescription.md'
_CONTINUITY = 'household/treatment-start.md'
_PATHS = (_INDEX, _PRESCRIPTION, _CONTINUITY)
_ARMS = ('current_v3', 'targeted_v3_projection', 'single_loop')
_HANDOFF_CASES = {'missing_start', 'conflicting', 'denied'}


def compare_safety(case_name: str, *, scripted_answer: dict) -> dict:
    """Run each real processor with the same synthetic, caller-supplied script.

    Source records are serialized without logging a read; processor read events
    trigger the separate fixture audit. File bytes are compared to the audited
    observations before any source or answer outcome may pass. Handoff/denial
    cases remain unavailable until real question/delivery/authority integration
    exists. No result qualifies production authorization or user-visible delivery.
    """
    if not isinstance(scripted_answer, dict):
        raise ValueError('structured_scripted_answer_required')
    case = next((item for item in CASES if item.name == case_name), None)
    if case is None or case_name in _HANDOFF_CASES:
        reason = 'unknown_case' if case is None else 'question_or_denial_delivery_not_integrated'
        return {'case': case_name, 'arms': {arm: {'availability': 'unavailable', 'passed': None,
                'reason': reason} for arm in _ARMS}, 'decision_qualified': False,
                'delivery': 'unassessed'}

    answer = json.dumps(scripted_answer, ensure_ascii=False)
    arms = {}
    for arm in _ARMS:
        fixture = PrescriptionFixture(case)
        # Only synthetic source contents enter the processor; the fixture's
        # separately specified expected outcomes do not enter these files.
        source = fixture._sources
        files = {_INDEX: json.dumps(source['index'], sort_keys=True),
                 _PRESCRIPTION: json.dumps(source['prescription'], sort_keys=True),
                 _CONTINUITY: json.dumps({'start': source['start'], 'status': source['status']}, sort_keys=True)}
        case_input = {'id': _CASE_ID, 'goal': task_request(case), 'fixture': {'files': files}}
        events = []
        discrepancies = []
        with TemporaryDirectory(prefix='prescription-safety-') as temporary:
            root = Path(temporary)
            for path, contents in files.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents, encoding='utf-8')

            def emit(event):
                events.append(event)
                if event.get('kind') != 'source_read':
                    return
                path = event.get('source')
                sources = {_INDEX: ('index',), _PRESCRIPTION: ('prescription',),
                           _CONTINUITY: ('start', 'status')}.get(path)
                if sources is None:
                    discrepancies.append('unexpected_source')
                    return
                observations = {name: fixture.read(name) for name in sources}
                if any(row.get('status') != 'ok' for row in observations.values()):
                    discrepancies.append('source_read_failed')
                    return
                value = (observations[sources[0]]['value'] if len(sources) == 1 else
                         {name: observations[name]['value'] for name in sources})
                if (root / path).read_text(encoding='utf-8') != json.dumps(value, sort_keys=True):
                    discrepancies.append('fixture_file_mismatch')

            try:
                if arm == 'single_loop':
                    script = iter([*({'kind': 'tool', 'tool_id': 'native.fixture_read',
                                      'arguments': {'path': path}} for path in _PATHS),
                                   {'kind': 'answer', 'text': answer}])
                    adapter = PrescriptionCandidateAdapter(lambda _text: next(script))
                else:
                    adapter = PrescriptionV3Adapter(projection_enabled=arm == 'targeted_v3_projection',
                                                   scripted_answers={_CASE_ID: answer})
                run = adapter(case_input, root, emit)
                try:
                    parsed = json.loads(run.response)
                except (ValueError, TypeError):
                    parsed = None
                fixture.observe_response(parsed)
                oracle = evaluate(case, fixture, parsed)
                passed = run.status == 'completed' and oracle['passed'] and not discrepancies
                arms[arm] = {'availability': 'available', 'passed': bool(passed),
                             'processor_status': run.status, 'response': parsed,
                             'oracle': oracle, 'source_discrepancies': discrepancies,
                             'events': events, 'source_audit': fixture.audit(),
                             'delivery': 'unassessed'}
            except Exception as exc:
                arms[arm] = {'availability': 'error', 'passed': None, 'reason': type(exc).__name__,
                             'events': events, 'source_audit': fixture.audit(), 'delivery': 'unassessed'}
    return {'case': case_name, 'arms': arms, 'decision_qualified': False, 'delivery': 'unassessed',
            'limitations': ['Preselected actions and answers do not demonstrate autonomous reasoning.',
                            'Source audit and permissions are synthetic, not production authority.',
                            'No user delivery, clinical validation, live provider usage or decision gate is assessed.']}
