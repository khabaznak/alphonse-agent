from alphonse.agent_v2.evaluation.dojo_attempt_summary import summarize


def test_incomplete_attempt_does_not_invent_a_result():
    summary = summarize([{'event': 'started', 'case': 'synthetic'},
                         {'event': 'model_failed', 'category': 'provider_unavailable'}])
    assert summary['journal_complete'] is False
    assert summary['effect_passed'] is None
    assert summary['response_semantics'] == summary['recovery_semantics'] == 'unassessed'
    assert summary['provider_failures'] == ['provider_unavailable']


def test_effect_and_response_assessments_remain_distinct():
    summary = summarize([{'event': 'model_decision', 'tool_id': 'native.bash',
                          'usage': {'input_tokens': 20, 'output_tokens': 2}},
                         {'event': 'bash_started'}, {'event': 'bash_finished', 'status': 'completed'},
                         {'event': 'finished', 'effect_passed': True, 'response_semantics': 'unassessed'}])
    assert summary['tool_choices'] == {'native.bash': 1}
    assert summary['bash_processes_started'] == 1
    assert summary['model_usage_observed'] == {'input_tokens': 20, 'output_tokens': 2}
    assert summary['effect_passed'] is True
    assert summary['response_semantics'] == 'unassessed'
