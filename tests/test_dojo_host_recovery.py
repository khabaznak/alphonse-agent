from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation import dojo_host_recovery as scenario


def test_disposable_recovery_fixture_is_independently_checked(tmp_path):
    workspace = scenario.seed(tmp_path)
    goal = scenario.goal(workspace)
    assert str(workspace / scenario.REPORT) in goal
    assert scenario.NOTE.decode() not in goal
    observation = scenario.readback(workspace)
    result = scenario.grade(observation, 'The technician record reports 53.5 psi.')
    assert result['fixture_integrity'] == 'unchanged'
    assert result['answer_mentions_expected_reading']
    assert result['answer_mentions_expected_unit']
    assert result['answer_mentions_record']
    assert result['response_semantics'] == 'requires_manual_review'
    (workspace / scenario.ALTERNATIVE).write_text('changed')
    assert scenario.grade(scenario.readback(workspace), None)['fixture_integrity'] == 'changed'


def test_unavailable_readback_and_symlink_substitution_are_not_success(tmp_path):
    workspace = scenario.seed(tmp_path)
    note = workspace / scenario.ALTERNATIVE
    note.unlink()
    note.symlink_to('/etc/hosts')
    assert scenario.readback(workspace)['status'] == 'unavailable'
    assert scenario.grade(scenario.readback(workspace), None)['fixture_integrity'] == 'unavailable'


def test_fixture_cannot_overwrite_or_use_relative_root(tmp_path):
    scenario.seed(tmp_path)
    with pytest.raises(FileExistsError):
        scenario.seed(tmp_path)
    with pytest.raises(ValueError):
        scenario.seed(Path('relative'))


def test_literal_markers_do_not_claim_semantic_pass(tmp_path):
    workspace = scenario.seed(tmp_path)
    grade = scenario.grade(scenario.readback(workspace), 'Not 53.5 psi, according to the technician.')
    assert grade['answer_mentions_expected_reading']
    assert grade['response_semantics'] == 'requires_manual_review'
    assert 'effect_passed' not in grade
