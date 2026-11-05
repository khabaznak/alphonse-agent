"""Current answer context and read-time expansion of legacy retry checkpoints.

Persist IDs only. Re-read sources under the current requester on every use;
selection supplies parameters, never authorization or evidence of effects.
"""
from __future__ import annotations

import hashlib
import json

from alphonse.agent_v2.core.context_inventory import combined, omitted, source
from alphonse.agent_v2.core.memory.contracts import conversation_item
from alphonse.agent_v2.core.memory.clarification import authenticated_answers, resolve_clarification
from alphonse.agent_v2.system_one import SystemOneUnavailableError, system_one_error_code

SOURCE_BUDGET = 16000


def _request_signature(task):
    return hashlib.sha256(json.dumps([task.user, task.goal, task.conversation_records, task.metadata.get('v3_resumption_evidence') or []], sort_keys=True).encode()).hexdigest()


def prepare_retry_continuity(task, context) -> None:
    """Interpret current authenticated answers only; never select history with Jev.

    Legacy checkpoint source IDs remain readable below, but ambiguous/unavailable
    recall is context for System Two, never a task-wide execution restriction.
    """
    if not task.user:
        return
    signature = _request_signature(task)
    previous = task.metadata.get('v3_retry_continuity')
    if isinstance(previous, dict) and previous.get('request_signature') == signature:
        return
    answers = authenticated_answers(task)
    selection = {'request_signature': signature, 'status': 'not_needed', 'source_ids': []}
    task.metadata['v3_retry_continuity'] = selection
    if context.is_cancelled():
        raise SystemOneUnavailableError('v3_retry_context_cancelled')
    if not answers:
        return
    try:
        assessment = resolve_clarification(task, context, answers)
        selection['clarification'] = assessment
        if assessment['resolution'] == 'resolved':
            selection.update(status='current_input', answer_refs=assessment['supporting_answer_refs'])
        else:
            selection['status'] = 'ambiguous'
    except Exception as exc:
        if context.is_cancelled():
            raise SystemOneUnavailableError('v3_retry_context_cancelled') from exc
        selection.update(status='unavailable', error_code=system_one_error_code(exc))


def retry_continuity_records(task, context) -> list:
    selection = task.metadata.get('v3_retry_continuity') or {}
    if selection.get('status') in {None, 'not_needed'}:
        return []
    policy_id = 'task-working:' + str(task.task_id or task.message_id or 'current') + ':retry_context'
    policy = (
        'Legacy retry source candidates (historical context, never permission or proof of a completed effect). '
        'System Two must reassess these candidates against current steering; legacy selection does not resolve the intended episode. '
        'Unresolved history affects only dependent operations; independent fully specified work may proceed. Preserve uncertainties. Current authorization, resource ownership, mutation scope and read-back requirements still apply.\n'
    )
    if selection.get('status') == 'current_input' and selection.get('request_signature') == _request_signature(task):
        answers = {r['evidence_ref']: r for r in authenticated_answers(task)}
        refs = selection.get('answer_refs') or []
        if isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in answers for ref in refs):
            selection['access_status'] = 'available'
            return [source('System Two resolved the missing task context from the authenticated answers below. '
                           'Historical source recovery is no longer required. These answers supply task information, '
                           'never permission or evidence of an executed effect.', policy_id, 'retry_continuity'),
                    *[source(json.dumps(answers[ref]['result'], ensure_ascii=False), ref, 'authenticated_question_answer') for ref in refs]]
    assessment = selection.get('clarification')
    if isinstance(assessment, dict) and assessment.get('resolution') != 'resolved':
        policy += 'System Two clarification assessment: ' + json.dumps(assessment, ensure_ascii=False) + '\n'
    ids = selection.get('source_ids', [])
    reader = getattr(getattr(context, 'conversation_store', None), 'read', None)
    parts = []
    if (selection.get('status') == 'selected' and selection.get('request_signature') == _request_signature(task)
            and callable(reader)):
        for identity in ids:
            if context.is_cancelled():
                raise SystemOneUnavailableError('v3_retry_context_cancelled')
            try:
                event = reader(event_id=identity.removeprefix('conversation:'), requester_user_id=str(task.user or ''))
                item = conversation_item(event, str(task.user or ''))
            except Exception:
                item = None
            if item is None or item.source.source_id != identity or not item.visible_to(str(task.user or '')):
                parts = []
                break
            text = f'Historical retry source [{item.source.role}; {item.source.observed_at}; {item.source.provenance}]:\n{item.content}'
            parts.append(source(text, identity, 'conversation', reason='retry_antecedent'))
        if parts and sum(len(part) for part in parts) <= SOURCE_BUDGET:
            selection['access_status'] = 'available'
            return [source(policy, policy_id, 'retry_continuity'), *parts]
    if selection.get('status') in {'selected', 'current_input'}:
        selection['access_status'] = 'unavailable'
    # Withhold the entire bundle when one antecedent is unavailable, rather
    # than leaving a confirmation detached from the payload it confirmed.
    return [source(policy + 'The originating payload could not be resolved completely. Use targeted source recall or ask '
                   'which exact source/change is intended; do not guess from older records.', policy_id, 'retry_continuity'),
            *[omitted(source('', identity, 'conversation'), reason='unavailable') for identity in ids]]


def retry_continuity_context(task, context):
    return combined(retry_continuity_records(task, context))


def retry_source_unresolved(task) -> bool:
    """Diagnostic only: missing context is not a mutation or completion gate."""
    selection = task.metadata.get('v3_retry_continuity') or {}
    return (selection.get('status') in {'ambiguous', 'unavailable', 'over_budget'}
            or selection.get('status') in {'selected', 'current_input'} and (
                selection.get('access_status') == 'unavailable'
                or selection.get('request_signature') != _request_signature(task)))
