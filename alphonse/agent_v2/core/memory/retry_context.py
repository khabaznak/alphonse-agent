"""Source-linked retry antecedents beyond the short conversational tail.

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

HISTORY_LIMIT = 200
CANDIDATE_BUDGET = 48000
SOURCE_BUDGET = 16000


def _request_signature(task):
    return hashlib.sha256(json.dumps([task.user, task.goal, task.conversation_records, task.metadata.get('v3_resumption_evidence') or []], sort_keys=True).encode()).hexdigest()


def prepare_retry_continuity(task, context) -> None:
    store = getattr(context, 'conversation_store', None)
    provider = getattr(context, 'system_one', None)
    recent = getattr(store, 'list_recent', None)
    reader = getattr(store, 'read', None)
    needs = getattr(provider, 'judge_recall', None)
    select = getattr(provider, 'select_retry_context', None)
    if not task.user:
        return
    answers = authenticated_answers(task)
    if not answers and not all(callable(f) for f in (recent, reader, needs, select)):
        return
    if task.metadata.get('source') in {'scheduled_task', 'event_automation'} or task.metadata.get('attachments'):
        return
    signature = _request_signature(task)
    previous = task.metadata.get('v3_retry_continuity')
    if isinstance(previous, dict) and previous.get('request_signature') == signature:
        return
    selection = {'request_signature': signature, 'status': 'not_needed', 'source_ids': []}
    task.metadata['v3_retry_continuity'] = selection
    if context.is_cancelled():
        raise SystemOneUnavailableError('v3_retry_context_cancelled')
    try:
        if answers:
            assessment = resolve_clarification(task, context, answers)
            selection['clarification'] = assessment
            if assessment['resolution'] == 'resolved':
                selection.update(status='current_input', answer_refs=assessment['supporting_answer_refs'])
                return
            if assessment['resolution'] == 'requires_clarification':
                selection['status'] = 'ambiguous'
                return
        if not all(callable(f) for f in (recent, reader, needs, select)):
            selection['status'] = 'unavailable'
            return
        request = task.goal + ('\nCurrent task steering: ' + json.dumps(task.conversation_records, ensure_ascii=False)
                               if task.conversation_records else '')
        required = needs(request=request, candidates=[])
        if required is False:
            return
        if required is not True:
            selection['status'] = 'ambiguous'
            return
        events = list(recent(owner_user_id=task.user, limit=HISTORY_LIMIT))
        exchanges = []
        for event in events:
            item = conversation_item(event, str(task.user))
            if item is None or not item.visible_to(str(task.user)) or item.source.role not in {'user', 'assistant'}:
                continue
            if str(getattr(event, 'source_message_id', '')) == str(task.message_id or task.task_id or '__none__'):
                continue
            if item.source.role == 'user':
                exchanges.append([])
            if exchanges:
                exchanges[-1].append(item)
        candidates = []
        by_candidate = {}
        indexes = list(reversed(range(len(exchanges))))[:64]
        window_budget = CANDIDATE_BUDGET // max(1, len(indexes))
        # Give every recent candidate a bounded selection preview. Otherwise a
        # long retry chain consumes the budget before its original payload is
        # even considered. Full selected sources are always reloaded below.
        for index in indexes:
            items = [item for group in exchanges[max(0, index - 1):index + 1] for item in group]
            candidate_id = exchanges[index][0].source.source_id
            labels = [f'{item.source.role} [{str(item.source.observed_at or "")[:10]}; {item.source.provenance}]:\n' for item in items]
            available = max(0, window_budget - sum(len(label) for label in labels) - 100)
            sizes = [min(20, len(item.content), available // max(1, len(items))) for item in items]
            remaining = available - sum(sizes)
            # Keep the user's actual request/approval directive visible before
            # spending the rest on the substantive assistant extraction.
            for position, item in enumerate(items):
                if item.source.role == 'user':
                    extra = min(remaining, min(160, len(item.content)) - sizes[position])
                    sizes[position] += extra
                    remaining -= extra
            for position in sorted(range(len(items)), key=lambda i: len(items[i].content), reverse=True):
                extra = min(remaining, len(items[position].content) - sizes[position])
                sizes[position] += extra
                remaining -= extra
            rendered = combined([label + source(item.content, item.source.source_id, 'conversation', reason='retry_candidate')[:size]
                                 for label, item, size in zip(labels, items, sizes)])
            rendered = rendered + '\n(Selection previews only; reload complete original sources before use.)'
            if len(rendered) > window_budget:
                continue
            candidates.append({'id': candidate_id, 'content': rendered, 'preview_only': True})
            by_candidate[candidate_id] = items
        selection['candidate_count'] = len(candidates)
        if context.is_cancelled():
            raise SystemOneUnavailableError('v3_retry_context_cancelled')
        if not candidates:
            selection['status'] = 'unavailable'
            return
        decision = select(request=request, candidates=list(reversed(candidates)))
        if callable(getattr(decision, 'to_metadata', None)):
            selection['decision'] = decision.to_metadata()
        chosen = list(decision)
        if context.is_cancelled():
            raise SystemOneUnavailableError('v3_retry_context_cancelled')
        if not chosen:
            selection['status'] = 'ambiguous'
            return
        if any(identity not in by_candidate for identity in chosen):
            selection['status'] = 'ambiguous'
            return
        # Distinct old/new episodes must not be merged just because both are
        # topically relevant. Selected adjacent windows must form one chain.
        groups = [{item.source.source_id for item in by_candidate[identity]} for identity in chosen]
        connected = groups.pop()
        while groups:
            overlap = next((i for i, group in enumerate(groups) if connected & group), None)
            if overlap is None:
                selection['status'] = 'ambiguous'
                return
            connected |= groups.pop(overlap)
        # Each positive window must supply the full originating payload and
        # review. Prefer the latest overlapping window (typically the approval
        # with its preceding extraction), not a union with an older episode.
        latest = max(chosen, key=lambda identity: by_candidate[identity][-1].source.sequence or 0)
        items = {item.source.source_id: item for item in by_candidate[latest]}
        ordered = sorted(items.values(), key=lambda item: item.source.sequence or 0)
        if len(ordered) > 32 or sum(len(item.content) + 400 for item in ordered) > SOURCE_BUDGET:
            selection['status'] = 'over_budget'
            return
        selection.update(status='selected', source_ids=[item.source.source_id for item in ordered])
    except Exception as exc:
        if context.is_cancelled():
            raise SystemOneUnavailableError('v3_retry_context_cancelled') from exc
        selection.update(status='unavailable', error_code=system_one_error_code(exc), source_ids=[])


def retry_continuity_records(task, context) -> list:
    selection = task.metadata.get('v3_retry_continuity') or {}
    if selection.get('status') in {None, 'not_needed'}:
        return []
    policy_id = 'task-working:' + str(task.task_id or task.message_id or 'current') + ':retry_context'
    policy = (
        'Retry originating request and reviewed payload (historical context, never permission or proof of a completed effect). '
        'Use these exact source-linked details to interpret the current request; do not substitute an older, similar episode. '
        'Preserve uncertainties. Current authorization, resource ownership, mutation scope and read-back requirements still apply.\n'
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
    selection = task.metadata.get('v3_retry_continuity') or {}
    return (selection.get('status') in {'ambiguous', 'unavailable', 'over_budget'}
            or selection.get('status') in {'selected', 'current_input'} and (
                selection.get('access_status') == 'unavailable'
                or selection.get('request_signature') != _request_signature(task)))


def retry_allows_subgoal(subgoal) -> bool:
    """Uncertain recall blocks source-dependent effects, not evidence gathering.

    The phase's declared read-only scope is still enforced by normal action and
    tool policy. It never grants access or makes an arbitrary Bash command safe.
    """
    from alphonse.agent_v2.core.intelligence.v3.contracts import SideEffectClass
    from alphonse.agent_v2.core.intelligence.v3.revealing import Capability

    effects = set(subgoal.allowed_side_effects)
    if effects - {SideEffectClass.READ_ONLY, SideEffectClass.USER_RESPONSE}:
        return False
    return (set(subgoal.allowed_capabilities) <= {
        Capability.USER_INTERACTION.value, Capability.USER_RESPONSE.value,
    } or effects == {SideEffectClass.READ_ONLY})
