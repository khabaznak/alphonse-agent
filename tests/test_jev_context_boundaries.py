"""Synthetic oversized tool observations and global Jev transport contracts."""
from copy import deepcopy
import json

import pytest

from alphonse.agent_v2.jev_context import PER_QUESTION_BYTES, PER_REQUEST_BYTES, encoded_size, tactical_review_state
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings, TypeSafeSystemOneClient


def large_search():
    hits = [{'source_id': f'conversation:fixture-{index}', 'source_type': 'conversation',
             'excerpt': ('Complete synthetic source passage ' + str(index) + ' ') * 25,
             'truncated': False, 'knowledge': {'content': ('Complete synthetic source passage ' + str(index) + ' ') * 25, 'status': 'active'}}
            for index in range(30)]
    return {'status': 'ok', 'results': hits, 'total_matches': 30, 'next_offset': None,
            'matches_markdown': 'DUPLICATE MARKDOWN ' * 5000,
            'recall': {'results': deepcopy(hits)}, 'knowledge': ['DUPLICATE KNOWLEDGE ' * 1000]}


def provider_with_positive_transport():
    calls = []
    def transport(url, key, payload, timeout):
        calls.append(payload)
        assert encoded_size({'state': payload['state'], 'questions': payload['questions']}) <= PER_REQUEST_BYTES
        for name, question in payload['questions'].items():
            assert encoded_size({'state': payload['state'], 'questions': {name: question}}) <= PER_QUESTION_BYTES
        return {'model': 'fixture', 'answers': {name: {'type': 'noul', 'noul': .99} for name in payload['questions']},
                'usage': {'input_tokens': 100}}
    return JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport), calls


def test_oversized_memory_search_is_canonical_bounded_and_cannot_complete_or_approve_retry():
    provider, calls = provider_with_positive_transport()
    result = large_search(); original = deepcopy(result)
    action = {'action_id': 'current', 'tool_id': 'native.search_memory', 'status': 'success', 'result': result}
    review = provider.evaluate_tactical_progress(goal='Find the latest fixture record',
        phase={'objective': 'Find and report', 'subgoals': []}, subgoal={'objective': 'Identify the exact source'},
        action=action, execution_log=[{'evidence_ref': 'tactical-action:current', **action}],
        task_context='UNRELATED BROAD PLANNING CONTEXT ' * 10000)
    assert len(calls) == 1
    state = calls[0]['state']
    assert 'task_context' not in state
    assert 'DUPLICATE MARKDOWN' not in json.dumps(state)
    first = state['current_action']['result']['results'][0]
    assert 'content' not in first['knowledge']
    assert first['knowledge']['status'] == 'active'
    assert json.dumps(state).count(first['excerpt']) == 1
    assert state['tool_execution_log'] == []
    assert state['context_coverage']['omitted_current_hits'] > 0
    assert not review.context_complete and not review.complete and not review.retry_approved
    assert review.answers['runtime_declared_subgoal_complete'] == .99  # Never quietly rewrite model probabilities.
    assert result == original


def test_targeted_read_after_large_search_can_complete_using_complete_current_observation():
    provider, calls = provider_with_positive_transport()
    result = {'items': [{'source_id': 'conversation:fixture-1', 'content': 'Latest fixture record dated October 30.'}]}
    review = provider.evaluate_tactical_progress(goal='Find latest record', phase={},
        subgoal={'objective': 'Identify source'}, action={'action_id': 'read', 'tool_id': 'native.read_memory_source',
            'status': 'success', 'result': result}, execution_log=[{'evidence_ref': 'search', 'tool_id': 'native.search_memory',
                'status': 'success', 'result': large_search()}])
    assert review.context_complete and review.complete
    assert calls[0]['state']['current_action']['result'] == result
    assert calls[0]['state']['tool_execution_log'][0]['result_omitted']
    assert review.context_coverage['omitted_prior_result_count'] == 1


def test_large_opaque_result_requires_a_targeted_observation_instead_of_silent_slicing():
    state = tactical_review_state('Read fixture', {}, {'objective': 'Read specific row'},
        {'tool_id': 'native.bash', 'status': 'success', 'result': {'stdout': 'LARGE RESULT ' * 10000}}, [])
    assert not state['context_coverage']['current_result_complete']
    assert state['current_action']['result']['omitted'] is True
    assert 'LARGE RESULT' not in json.dumps(state)


@pytest.mark.parametrize('purpose', ['admission', 'admission_curation', 'completion_review', 'act_judgments',
                                   'evaluate_tactical_progress', 'recall_selection', 'review_memory_candidates'])
def test_shared_boundary_blocks_oversized_states_before_network_for_any_decision(purpose):
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'),
        transport=lambda *args: pytest.fail('Oversized input must not be sent'))
    with pytest.raises(ValueError, match='system_one_context_budget_exceeded'):
        provider._evaluate(purpose=purpose, state={'required': '漢' * 12000},
            questions={'question': {'type': 'noul', 'instructions': 'Is the source present?'}})


def test_large_independent_question_sets_are_partitioned_without_losing_answers():
    provider, calls = provider_with_positive_transport()
    questions = {f'q-{index}': {'type': 'noul', 'instructions': 'Atomic synthetic judgment. ' * 120}
                 for index in range(45)}
    response = provider._evaluate(purpose='fixture_parallel', state={'source': 'fixture'}, questions=questions)
    assert len(calls) > 1
    assert set(response['answers']) == set(questions)
    assert sum(len(call['questions']) for call in calls) == len(questions)
    assert response['usage']['input_tokens'] == len(calls) * 100


def test_single_oversized_choice_is_not_silently_pruned():
    client = TypeSafeSystemOneClient(api_url='https://api.typesafe.ai/v1/systemone', api_key='fixture', model='fixture',
        transport=lambda *args: pytest.fail('Cannot change Choice options to fit the context'))
    with pytest.raises(ValueError, match='system_one_context_budget_exceeded'):
        client.evaluate(state={'source': 'fixture'}, questions={'choice': {'type': 'choice', 'instructions': 'Select the source',
            'criteria': {'a': 'Complete source A ' * 4000, 'b': 'Complete source B'}}})


def test_large_completion_review_partitions_whole_evidence_without_duplicate_renderings():
    provider, calls = provider_with_positive_transport()
    entries = [{'evidence_ref': f'fixture:{index}', 'status': 'success', 'tool_id': 'native.bash',
                'result': {'text': 'SOURCE PASSAGE ' * 200}} for index in range(12)]
    reviewed = provider.evaluate(contract={'criteria': [{'id': 'ac', 'statement': 'A fixture source is retrieved', 'status': 'pending'}]},
        phase={'objective': 'Read fixture'}, evidence={'entries': entries}, task_context='IRRELEVANT OLD CONTEXT ' * 5000)
    assert reviewed.updates
    assert all('ordered_verified_evidence' not in call['state'] and 'task_context' not in call['state'] for call in calls)
    supplied_refs = set().union(*(set(call['state']['verified_evidence']) for call in calls))
    assert supplied_refs == {entry['evidence_ref'] for entry in entries}
