import pytest

from alphonse.agent_v2.evaluation.planning_multifield import run_case


@pytest.mark.parametrize('scenario', ['normal', 'changed', 'denied', 'cancel', 'uncertain',
                                      'cancel_during', 'changed_during', 'denied_during', 'restart'])
@pytest.mark.parametrize('batch', [False, True])
def test_tactical_shapes_preserve_fixture_effect_and_authorization_guards(scenario, batch):
    result = run_case(scenario, batch)
    assert result['independent_oracle_pass'], result
    if scenario in {'normal', 'uncertain', 'restart'}:
        assert result['verified'] and result['submission_count'] == 1
    else:
        assert not result['verified'] and result['submission_count'] == 0


def test_fixture_batch_reduces_calls_without_dropping_readback():
    baseline = run_case('normal', False)
    batch = run_case('normal', True)
    assert baseline['operations'] == batch['operations'] == ['inspect', 'fill', 'fill', 'submit', 'verify']
    assert (baseline['composition_calls'], batch['composition_calls']) == (5, 3)
    assert (baseline['jev_reviews'], batch['jev_reviews']) == (5, 3)


def test_failed_model_response_is_recorded_without_claiming_completion():
    class BrokenProvider:
        def generate_json(self, request):
            raise ValueError('openai_codex_invalid_json')
    result = run_case('normal', False, inference=BrokenProvider())
    assert result['outcome'] == 'error'
    assert result['failure_code'] == 'openai_codex_invalid_json'
    assert not result['independent_oracle_pass']
    assert result['submission_count'] == 0
    assert result['composition_calls'] == 1


def test_live_model_wrapper_disables_host_and_connected_tools(monkeypatch):
    import os
    import sys
    from pathlib import Path
    from alphonse.agent_v2.evaluation.planning_multifield_live import _model_only_wrapper_source
    captured = []
    monkeypatch.setattr(sys, 'argv', ['wrapper', 'exec', '--ephemeral'])
    monkeypatch.setattr(os, 'execv', lambda binary, args: captured.append((binary, args)))
    source = _model_only_wrapper_source(Path('/synthetic/codex'))
    exec(compile(source, '<synthetic-wrapper>', 'exec'), {})
    binary, args = captured[0]
    assert binary == '/synthetic/codex'
    for feature in ('shell_tool', 'unified_exec', 'apps', 'browser_use', 'computer_use', 'code_mode_host'):
        assert any(args[i:i + 2] == ['--disable', feature] for i in range(len(args) - 1))
    assert 'mcp_servers={}' in args
    assert 'read-only' in args
    assert 'project_doc_max_bytes=0' in args


@pytest.mark.parametrize('batch', [False, True])
def test_restart_preserves_observation_and_does_not_replay_it(batch):
    result = run_case('restart', batch)
    assert result['restarted'] and result['independent_oracle_pass']
    assert result['operations'] == ['inspect', 'fill', 'fill', 'submit', 'verify']
