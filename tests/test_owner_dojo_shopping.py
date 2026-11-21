from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation.owner_dojo_shopping import ScriptedProvider, _capability_outcome, run


def test_scripted_fixture_has_one_fixed_container_command_and_no_model_call():
    provider = ScriptedProvider()
    first = provider.generate_json(SimpleNamespace(prompt='available: native.bash'))
    assert provider.saw_bash_catalog
    assert first.json_value['tool_id'] == 'native.bash'
    assert '/workspace/household' in first.json_value['arguments']['command']
    assert '/workspace/shopping-list.json' in first.json_value['arguments']['command']
    second = provider.generate_json(SimpleNamespace(prompt='fixture observation'))
    assert second.json_value['kind'] == 'answer'
    assert provider.calls == 2


def test_usage_block_before_bash_does_not_grade_an_unrun_tool():
    decision = [{'kind': 'tool', 'tool_id': 'native.bash'}]
    result = _capability_outcome('failed', 'usage_limit_exceeded', decision, [], False)
    assert result == {'capability_outcome': 'unassessed_blocked_before_bash', 'task_passed': None}
    assert _capability_outcome('failed', 'usage_limit_exceeded', decision, [{}], False)['task_passed'] is False


def test_synthetic_authority_denial_cannot_silently_run_without_the_service():
    with pytest.raises(ValueError, match='denial_requires_synthetic_authority'):
        run(image_id='', docker_socket='', emit=lambda _: None, deny_authority_dispatch=True)


def test_provider_failure_is_inconclusive_even_if_the_effect_is_missing():
    result = _capability_outcome('failed', 'model_unavailable', [], [], False, ['openai_codex_exec_failed'])
    assert result == {'capability_outcome': 'unassessed_provider_failure', 'task_passed': None}


def test_authority_denial_before_dispatch_does_not_grade_an_unrun_model_action():
    result = _capability_outcome('failed', 'needs_attention', [], [], False, (),
                                 [{'event': 'authority_denied', 'stage': 'action_start'}])
    assert result == {'capability_outcome': 'unassessed_authority_denied_before_bash', 'task_passed': None}
