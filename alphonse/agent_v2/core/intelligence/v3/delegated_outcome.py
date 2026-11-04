"""Deterministic completion gates over atomic judgments and actual observations."""
from __future__ import annotations

import math

from alphonse.agent_v2.outcome_judgments import OUTCOME_QUESTIONS


def outcome_assessment(answers, evidence, *, yes=.8, no=.2):
    """Return a receipt; uncertainty cannot become permission or success."""
    valid = all(type(answers.get(key)) in {int, float}
                and math.isfinite(answers[key]) and 0 <= answers[key] <= 1
                for key in OUTCOME_QUESTIONS)
    if not valid:
        return {'status': 'unresolved', 'fulfilled': False, 'reason': 'missing_or_invalid_outcome_judgment'}
    positive = lambda key: answers[key] >= yes
    negative = lambda key: answers[key] <= no
    operational = [row for row in evidence if row.get('status') == 'success'
                   and row.get('tool_id') and row['tool_id'] not in {'native.respond', 'native.ask_question'}
                   and isinstance(row.get('result'), dict)
                   and ('verification' not in row['result']
                        or isinstance(row['result']['verification'], dict)
                        and row['result']['verification'].get('status') == 'verified')]
    # A random successful read is not proof of schedule activation. The native
    # scheduling tool returns this receipt only after reading the stored record.
    schedules = [row for row in operational if row['tool_id'] == 'native.scheduled_task'
                 and row['result'].get('scheduled_task_id') and row['result'].get('status') == 'active'
                 and row['result'].get('next_run_at')
                 and isinstance(row['result'].get('schedule'), dict)
                 and row['result']['schedule'].get('kind') == 'rrule'
                 and row['result'].get('verification') == {'status': 'verified', 'source': 'schedule_store_readback'}]
    reason = ''
    if not positive('mission_outcome_is_fulfilled'):
        reason = 'delegated_outcome_unverified'
    elif not positive('capability_claims_are_supported'):
        reason = 'capability_claims_unverified'
    elif not positive('lookup_claims_are_supported'):
        reason = 'lookup_coverage_unverified'
    elif not negative('mission_requires_execution') and not operational:
        reason = 'execution_evidence_required'
    elif not negative('mission_requires_recurring_setup') and not schedules:
        reason = 'schedule_persistence_required'
    fulfilled = not reason
    status = ('fulfilled' if fulfilled else 'blocked' if positive('blocker_is_supported') else
              'intermediate_progress' if positive('work_is_useful_progress') else 'unresolved')
    return {'status': status, 'fulfilled': fulfilled, 'reason': reason,
            'capability_assessment_supported': positive('mission_is_capability_only') and positive('capability_claims_are_supported'),
            'supported_blocker': positive('blocker_is_supported'),
            'useful_progress': positive('work_is_useful_progress'),
            'operational_evidence_refs': [row.get('evidence_ref') for row in operational],
            'schedule_evidence_refs': [row.get('evidence_ref') for row in schedules]}
