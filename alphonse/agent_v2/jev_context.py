"""Bounded Jev decision inputs; raw tool results remain in the checkpoint."""
from __future__ import annotations

from copy import deepcopy
import json

PER_QUESTION_BYTES = 28000
PER_REQUEST_BYTES = 56000
TACTICAL_STATE_BYTES = 20000


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False).encode('utf-8'))


def question_batches(state, questions):
    """Split independent questions, never their criteria/options or the state."""
    for name, question in questions.items():
        if encoded_size({'state': state, 'questions': {name: question}}) > PER_QUESTION_BYTES:
            raise ValueError('system_one_context_budget_exceeded')
    batches, batch = [], {}
    for name, question in questions.items():
        candidate = {**batch, name: question}
        if batch and encoded_size({'state': state, 'questions': candidate}) > PER_REQUEST_BYTES:
            batches.append(batch)
            batch = {}
        batch[name] = question
    if batch:
        batches.append(batch)
    return batches


def canonical_result(tool_id, result):
    if tool_id != 'native.search_memory' or not isinstance(result, dict) or not isinstance(result.get('results'), list):
        return deepcopy(result)
    # These are repeated renderings of the same hits, not additional evidence.
    value = {key: deepcopy(result[key]) for key in ('status', 'schema_version', 'query', 'total_matches',
             'offset', 'next_offset', 'truncated', 'coverage', 'knowledge_status') if key in result}
    value['results'] = []
    for hit in result['results']:
        if not isinstance(hit, dict):
            continue
        row = {key: deepcopy(item) for key, item in hit.items() if key != 'knowledge'}
        knowledge = hit.get('knowledge')
        if isinstance(knowledge, dict):
            row['knowledge'] = {key: deepcopy(item) for key, item in knowledge.items()
                                if key != 'content' or item != hit.get('excerpt')}
        value['results'].append(row)
    return value


def tactical_review_state(goal, phase, subgoal, action, execution_log):
    """The current observation is primary; history is optional, never duplicated.

    A large search result keeps whole ranked hits. A large opaque result is
    withheld rather than silently sliced. A partial current observation cannot
    authorize completion, mutation readiness, a user-input handoff, or retries.
    System Two still receives the original result and can obtain narrower evidence.
    """
    current = {key: deepcopy(action[key]) for key in ('action_id', 'tool_id', 'arguments', 'status', 'error') if key in action}
    current['result'] = canonical_result(action.get('tool_id'), action.get('result'))
    value = {'goal': goal, 'stage_goal': deepcopy(subgoal),
        'strategic_plan': {'objective': phase.get('objective'), 'subgoals': [
            {key: row[key] for key in ('subgoal_id', 'objective', 'depends_on', 'allowed_side_effects') if key in row}
            for row in phase.get('subgoals', [])]},
        'current_action': current, 'tool_execution_log': [],
        'context_coverage': {'current_result_complete': True, 'omitted_prior_refs': [], 'omitted_prior_count': 0, 'omitted_current_hits': 0}}
    coverage = value['context_coverage']
    if encoded_size(value) > TACTICAL_STATE_BYTES - 2000:
        result = current['result']
        if action.get('tool_id') == 'native.search_memory' and isinstance(result, dict) and isinstance(result.get('results'), list):
            hits = result['results']; result['results'] = []
            for index, hit in enumerate(hits):
                result['results'].append(hit)
                if encoded_size(value) > TACTICAL_STATE_BYTES - 2000:
                    result['results'].pop()
                    coverage['current_result_complete'] = False
                    coverage['omitted_current_hits'] = len(hits) - index
                    break
        else:
            current['result'] = {'omitted': True, 'result_bytes': encoded_size(result),
                                 'reason': 'targeted_observation_required'}
            coverage['current_result_complete'] = False
    if encoded_size(value) > TACTICAL_STATE_BYTES - 1000:
        raise ValueError('system_one_context_budget_exceeded')
    action_id = action.get('action_id')
    for entry in reversed(list(execution_log or [])):
        if action_id and entry.get('evidence_ref') == 'tactical-action:' + action_id:
            continue
        row = {key: deepcopy(entry[key]) for key in ('evidence_ref', 'tool_id', 'status', 'error', 'phase_id', 'subgoal_id') if key in entry}
        result = canonical_result(entry.get('tool_id'), entry.get('result'))
        if encoded_size(result) < 3000:
            row['result'] = result
        else:
            row['result_omitted'] = True
        value['tool_execution_log'].append(row)
        if encoded_size(value) > TACTICAL_STATE_BYTES - 1000:
            value['tool_execution_log'].pop()
            coverage['omitted_prior_count'] += 1
            if len(coverage['omitted_prior_refs']) < 16:
                coverage['omitted_prior_refs'].append(str(entry.get('evidence_ref') or 'prior'))
    value['tool_execution_log'].reverse()
    coverage['omitted_prior_result_count'] = sum(1 for row in value['tool_execution_log'] if row.get('result_omitted'))
    coverage['state_bytes'] = encoded_size(value)
    if encoded_size(value) > TACTICAL_STATE_BYTES:
        raise ValueError('system_one_context_budget_exceeded')
    return value
