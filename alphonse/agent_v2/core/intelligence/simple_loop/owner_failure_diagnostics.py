"""Bounded, actor-scoped descriptions of recorded owner-request failures.

These fixed codes do not expose exceptions, confirm side effects, or authorize
retries. A checkpoint is evidence of a blocker, not a complete audit log.
"""
from __future__ import annotations

import json
import re

_DESCRIPTIONS = {
    'time_limit': 'The request reached its time limit before completion.',
    'model_unavailable': 'The model step failed; its specific cause was not recorded.',
    'model_not_configured': 'The model was not configured.',
    'model_cli_missing': 'The configured model command was not found.',
    'model_auth_unverified': 'The model provider sign-in could not be verified.',
    'model_auth_required': 'The model provider required sign-in.',
    'model_cli_upgrade_required': 'The model command required an update.',
    'model_timeout': 'The model provider timed out.',
    'model_empty_response': 'The model returned an empty response.',
    'model_response_invalid': 'The model response had an invalid format.',
    'input_reservation_limit': 'The model input exceeded its reserved limit.',
    'output_reservation_limit': 'The model output exceeded its reserved limit.',
    'usage_unconfirmed': 'Model usage could not be verified.',
    'usage_limit_exceeded': 'The model exceeded its request usage limit.',
    'model_call_limit': 'The next model call could not be reserved within the request limit.',
    'input_budget_limit': 'The next model input could not be reserved within the request budget.',
    'output_budget_limit': 'The next model output could not be reserved within the request budget.',
    'action_limit': 'The request reached its action limit.',
    'source_unavailable': 'A needed source was unavailable or access was denied.',
    'source_changed': 'A source changed or access was revoked.',
    'source_incomplete': 'A source could not be verified as complete.',
    'unsupported_operation': 'A requested operation was unavailable.',
    'answer_unverified': 'The proposed answer could not be verified; the specific reason was not recorded.',
    'clarification_unverified': 'The clarification did not pass its check.',
    'clarification_review_unavailable': 'The clarification check was unavailable.',
    'needs_attention': 'The recorded checkpoint did not specify a recognized cause.',
}

def public_failure_code(error: object, *, cancelled: bool = False) -> str:
    if cancelled:
        return 'cancel_requested'
    return {
        'candidate_deadline_exceeded': 'time_limit',
        'BudgetExhausted:budget_elapsed_exhausted': 'time_limit',
        'candidate_model_unavailable': 'model_unavailable',
        'candidate_model_not_configured': 'model_not_configured',
        'candidate_model_cli_missing': 'model_cli_missing',
        'candidate_model_auth_unverified': 'model_auth_unverified',
        'candidate_model_auth_required': 'model_auth_required',
        'candidate_model_cli_upgrade_required': 'model_cli_upgrade_required',
        'candidate_model_timeout': 'model_timeout',
        'candidate_model_empty_response': 'model_empty_response',
        'candidate_model_response_invalid': 'model_response_invalid',
        'candidate_input_reservation_too_small': 'input_reservation_limit',
        'candidate_output_reservation_too_small': 'output_reservation_limit',
        'candidate_model_usage_unknown_or_overrun': 'usage_unconfirmed',
        'candidate_model_usage_overrun': 'usage_limit_exceeded',
        'BudgetExhausted:budget_model_usage_unresolved': 'usage_unconfirmed',
        'BudgetExhausted:budget_exhausted:actions': 'action_limit',
        'BudgetExhausted:budget_exhausted:model_calls': 'model_call_limit',
        'BudgetExhausted:budget_exhausted:input_tokens': 'input_budget_limit',
        'BudgetExhausted:budget_exhausted:output_tokens': 'output_budget_limit',
        'candidate_read_denied': 'source_unavailable',
        'candidate_read_revoked': 'source_unavailable',
        'candidate_retry_deferred': 'source_unavailable',
        'candidate_evidence_changed_or_revoked': 'source_changed',
        'candidate_observation_incomplete': 'source_incomplete',
        'candidate_requires_read_only_native_tool': 'unsupported_operation',
        'candidate_answer_unsupported': 'answer_unverified',
        'candidate_question_not_verified': 'clarification_unverified',
        'candidate_question_review_unavailable': 'clarification_review_unavailable',
    }.get(error, 'needs_attention') if isinstance(error, str) else 'needs_attention'


def recent_owner_failures(queue, events, *, actor: str, project_id: str, current_message_id: str) -> list[dict]:
    """Return at most three allowlisted blockers for preceding trusted owner turns.

    Source IDs and checkpoints are independently actor/project matched. A
    terminal blocked checkpoint does not establish whether a dispatched effect
    occurred. Missing, malformed, foreign and nonterminal records are omitted.
    """
    if not isinstance(actor, str) or not actor or not isinstance(project_id, str):
        return []
    result = []
    with queue._connect() as conn:
        # A same-user Telegram or other-channel turn is not authorized to
        # inspect a trusted Desktop ingress merely by sharing an actor ID.
        current = conn.execute('''SELECT 1 FROM v2_trusted_owner_ingress o
            JOIN v2_conversation_events e ON e.event_id=o.history_event_id
            WHERE o.message_id=? AND o.owner_id=? AND o.history_version=1
              AND e.owner_user_id=? AND e.project_id=? AND e.role='user'
              AND e.source='desktop' AND e.source_message_id=?''',
            (current_message_id, actor, actor, project_id, 'inbound:' + current_message_id)).fetchone()
        if current is None:
            return []
        for event in reversed(events):
            source = getattr(event, 'source_message_id', '')
            if (getattr(event, 'role', None) != 'user' or not isinstance(source, str)
                    or not re.fullmatch(r'inbound:[0-9a-f]{32}', source)):
                continue
            message_id = source[8:]
            if message_id == current_message_id:
                continue
            row = conn.execute('''SELECT c.task_state_json, o.history_event_id, o.history_version FROM v2_task_checkpoints c
                JOIN v2_trusted_owner_ingress o ON o.message_id=c.task_id
                WHERE c.task_id=? AND c.owner_id=? AND o.owner_id=? AND coalesce(c.project_id,'')=?''',
                (message_id, actor, actor, project_id)).fetchone()
            if (row is None or row['history_version'] != 1
                    or row['history_event_id'] != getattr(event, 'event_id', None)
                    or getattr(event, 'source', None) != 'desktop'
                    or getattr(event, 'owner_user_id', None) != actor
                    or getattr(event, 'project_id', None) != project_id):
                continue
            try:
                task = json.loads(row['task_state_json'])
                from alphonse.agent_v2.core.messages.sqlite_queue import finalized_owner_history_matches
                if not finalized_owner_history_matches(conn, message_id, actor, task):
                    continue
                state = task['metadata']['single_loop_checkpoint']['state']
                if (task.get('user') != actor or task.get('message_id') != message_id
                        or task.get('task_id') != message_id or task.get('project_id', '') != project_id
                        or state.get('task_id') != message_id or state.get('actor') != actor
                        or state.get('status') != 'blocked'):
                    continue
                code = public_failure_code(state.get('blocker'))
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            # Internal legacy category names (e.g. model_unavailable) can
            # overstate the evidence. Give the model the fixed description,
            # not the code or raw blocker.
            result.append({'message_id': message_id, 'description': _DESCRIPTIONS[code],
                           'effects': 'not established by this diagnostic'})
            if len(result) == 3:
                break
    result.reverse()
    return result
