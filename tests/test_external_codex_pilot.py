"""No Codex subprocess or provider call is made by these tests."""
import json
from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation.external_codex_pilot import build_argv, inspect_trace, reference_response, response_schema


def lines(*events):
    return ''.join(json.dumps(event) + '\n' for event in events)


def test_reference_output_contracts_have_no_case_answers():
    for case in ('clock', 'traffic', 'prescription'):
        schema = response_schema(case)
        assert isinstance(schema, dict)
        assert 'FAKE-100' not in json.dumps(schema) and 'Asterin' not in json.dumps(schema)
    traffic = response_schema('traffic')
    assert traffic['type'] == 'object'
    assert [row['properties']['status']['const'] for row in traffic['properties']['result']['anyOf']] == ['verified', 'blocked']
    with pytest.raises(ValueError, match='unsupported_reference_case'):
        response_schema('gmail')


def test_private_response_validation_keeps_oracle_contracts_separate():
    assert reference_response('traffic', '{"result":{"status":"blocked","reason":"verification_required"}}') == {
        'status': 'blocked', 'reason': 'verification_required'}
    assert reference_response('traffic', '{"result":{"status":"verified","subject":"FAKE-1","ticket_ids":[]}}') == {
        'status': 'verified', 'subject': 'FAKE-1', 'ticket_ids': []}
    assert reference_response('traffic', '{"result":{"status":"verified","subject":"FAKE-1","ticket_ids":[],"extra":"x"}}') is None
    assert reference_response('clock', '{"local_time":"09:00"}') is None


def test_builds_fixed_stdin_command_without_execution(tmp_path):
    argv = build_argv(executable=Path('/usr/bin/codex'), workspace=tmp_path,
                      model='test-model', output_last_message=tmp_path / 'answer.json',
                      output_schema=tmp_path / 'schema.json')
    assert argv[-1] == '-'
    assert argv[1:5] == ['exec', '--json', '--ephemeral', '--ignore-user-config']
    assert argv[argv.index('--sandbox') + 1] == 'read-only'
    assert '--dangerously-bypass-approvals-and-sandbox' not in argv
    assert '--model' in argv and 'test-model' in argv
    with pytest.raises(ValueError, match='invalid_model'):
        build_argv(executable=Path('/usr/bin/codex'), workspace=tmp_path,
                   model='--bad', output_last_message=tmp_path / 'answer.json',
                   output_schema=tmp_path / 'schema.json')


def test_complete_trace_keeps_provider_usage_separate_from_effect_and_delivery():
    trace = lines({'type': 'thread.started'},
                  {'type': 'item.completed', 'item': {'type': 'command_execution', 'command': 'secret should not be output'}},
                  {'type': 'turn.completed', 'usage': {'input_tokens': 42, 'output_tokens': 7}})
    summary = inspect_trace(trace, final_text='{"local_time":"09:00"}', exit_code=0)
    assert summary['status'] == 'completed'
    assert summary['structured_response_available'] is True
    assert summary['reported_usage'] == {'input_tokens': 42, 'output_tokens': 7}
    assert summary['usage_source'] == 'codex_cli_reported' and summary['command_attempts'] == 1
    assert summary['fixture_effect'] == 'unassessed' and summary['decision_qualified'] is False
    assert 'secret should not be output' not in str(summary)


@pytest.mark.parametrize('trace,exit_code', [
    (lines({'type': 'thread.started'}), 0),
    (lines({'type': 'turn.failed'}), 0),
    (lines({'type': 'turn.completed'}, {'type': 'turn.completed'}), 0),
    (lines({'type': 'turn.completed'}, {'type': 'error'}), 0),
    (lines({'type': 'turn.completed'}) + '{', 0),
    (lines({'type': 'turn.completed'}), 1),
    (lines({'type': 'turn.completed'}), None),
])
def test_incomplete_or_failed_trace_cannot_produce_structured_success(trace, exit_code):
    result = inspect_trace(trace, final_text='{"status":"verified"}', exit_code=exit_code)
    assert result['status'] == 'incomplete_or_failed'
    assert result['structured_response_available'] is False and result['decision_qualified'] is False


def test_missing_usage_and_invalid_final_answer_are_unknown():
    trace = lines({'type': 'turn.completed', 'usage': {'input_tokens': 0, 'output_tokens': True}})
    result = inspect_trace(trace, final_text='a non-JSON answer', exit_code=0)
    assert result['reported_usage'] == {'input_tokens': 0, 'output_tokens': None}
    assert result['usage_source'] == 'partial_or_unknown'
    assert result['structured_response_available'] is False


def test_final_answer_is_not_disclosed_in_sanitized_summary():
    trace = lines({'type': 'turn.completed'})
    result = inspect_trace(trace, final_text='{"secret":"bearer-token-123"}', exit_code=0)
    assert result['structured_response_available'] is True
    assert 'bearer-token-123' not in repr(result)
