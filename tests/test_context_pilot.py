"""The pilot exercises product assembly and permits only disposable fixture effects."""
from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation.context_pilot import CASES, COMMAND, Fixture, projection_flag, run_arm


@pytest.mark.parametrize('case_id,message', CASES)
def test_both_context_arms_run_without_production_configuration(case_id, message, monkeypatch):
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', '/nonexistent/do-not-open')
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', 'original')
    for candidate in (False, True):
        row = run_arm(case_id, message, candidate=candidate)
        assert row['behavior_checks'] is None
        assert row['turns'] == []
        assert row['context_checks']['private_source_excluded']
    import os
    assert os.environ['ALPHONSE_V3_CONTEXT_PROJECTIONS'] == 'original'


def test_model_can_choose_a_fresh_fixture_read_and_answer():
    turns = iter([
        {'kind': 'tool', 'tool_id': 'fixture.read_sensor', 'arguments_json': '{"entity_id":"study_temperature"}', 'answer': ''},
        {'kind': 'answer', 'tool_id': '', 'arguments_json': '', 'answer': 'The study is 24 C.'},
    ])
    row = run_arm(*CASES[0], candidate=True, responder=lambda payload: next(turns))
    assert all(row['behavior_checks'].values())
    assert row['observations'][0]['result']['evidence_kind'] == 'fresh_fixture_observation'


def test_unregistered_or_unauthorized_effect_is_never_executed():
    for tool in ('native.send_message', 'fixture.unlock_door'):
        row = run_arm(*CASES[3], candidate=True, responder=lambda payload: {
            'kind': 'tool', 'tool_id': tool, 'arguments_json': '{}', 'answer': ''})
        assert row['error'] == 'PermissionError'
        assert row['observations'] == []
        assert not row['behavior_checks']['no_provider_or_tool_error']


def test_fixture_bash_rejects_arbitrary_command(tmp_path):
    case_id, message = CASES[4]
    fixture = Fixture(tmp_path, case_id, message)
    try:
        definition = fixture.registry.get_definition('native.bash')
        with pytest.raises(PermissionError, match='allowlist'):
            definition.callable({'command': 'touch ' + str(tmp_path / 'should-not-exist')})
        assert not (tmp_path / 'should-not-exist').exists()
        result = definition.callable({'artifact_id': 'artifact.pilot-sensor', 'command': COMMAND})
        assert result['exit_code'] == 0
        assert '23' in result['stdout']
        assert not any(t.tool_id.startswith('artifact.') for t in fixture.registry.list())
    finally:
        fixture.close()
