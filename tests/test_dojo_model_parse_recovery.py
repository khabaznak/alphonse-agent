"""Assert outcomes against the disposable command callback and owner IPC."""
import json

import pytest

from alphonse.agent_v2.evaluation.dojo_model_parse_recovery import (
    ANSWER, CASES, FIRST_FAILURE_USAGE, PRIVATE_OUTPUT, REJECTED_OUTPUT, run,
)


@pytest.mark.parametrize('case', CASES)
def test_timeout_followed_by_unreadable_model_response(case):
    report = run(case)
    unknown_usage = case == 'unknown_usage'
    assert report['command_dispatches'] == 1
    assert report['format_retries'] == (0 if unknown_usage else 1)
    assert len(report['model_calls']) == (2 if unknown_usage else 3)
    assert all(call['source'] == 'scripted' for call in report['model_calls'])
    assert report['model_calls'][0]['normalized_kind'] == 'tool'
    assert report['model_calls'][1]['parse_stage'] == 'arguments_invalid_json'
    assert all(call['saw_observation'] for call in report['model_calls'][1:])
    assert not any(call['saw_rejected_output'] for call in report['model_calls'])
    assert report['parse_failures']['arguments_invalid_json'] == {
        'reported_usage': {} if unknown_usage else FIRST_FAILURE_USAGE,
        'unknown_usage': sorted(FIRST_FAILURE_USAGE) if unknown_usage else [],
    }
    expected_stages = ['arguments_invalid_json']
    if case == 'malformed_again':
        assert report['model_calls'][-1]['parse_stage'] == 'outer_object'
        expected_stages.append('outer_object')
        assert report['parse_failures']['outer_object']['reported_usage'] == {'input_tokens': 220, 'output_tokens': 20}
    log = report['work_log']
    assert log['model_parse_stages'] == expected_stages
    assert log['events'] == [{'label': 'Local command', 'outcome': 'Command timed out; effects are unconfirmed'}]
    assert log['activity']['latest'] == {'tool_kind': 'local_command', 'outcome': 'timed_out', 'effects': 'unconfirmed'}
    assert PRIVATE_OUTPUT not in json.dumps(log)
    assert REJECTED_OUTPUT not in json.dumps(report)
    assert report['rejected_output_retained'] is False
    if case == 'recovers':
        assert report['model_calls'][-1]['normalized_kind'] == 'answer'
        assert report['status']['status'] == 'ready'
        assert report['answer'] == ANSWER
        assert report['acknowledgement'] == 'client_acknowledged'
    else:
        assert report['status'] == {'status': 'failed', 'failure_code': 'model_response_invalid'}
        assert log['status'] == 'blocked'
        assert report['answer'] is None
        if case == 'tries_to_replay':
            assert report['model_calls'][-1]['normalized_kind'] == 'tool'
