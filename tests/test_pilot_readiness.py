"""Presence-only preflight must never execute tools or grant authorization."""
from alphonse.agent_v2.evaluation.pilot_readiness import inspect_local


def test_present_files_never_authorize_or_qualify_a_live_run(tmp_path):
    fake = tmp_path / 'fake-executable'
    fake.write_text('synthetic, not a program')
    result = inspect_local(codex_path=fake, docker_path=fake)
    assert result['observed'] == {'codex_candidate': 'present_unverified',
                                  'docker_candidate': 'present_unverified',
                                  'docker_daemon': 'unverified'}
    assert result['dispatch_authorized'] is False and result['ready_to_run'] is False
    assert 'task_wide_nonbypassable_spend_control_unverified' in result['blockers']
    assert str(fake) not in str(result)


def test_missing_candidates_stay_unverified(tmp_path):
    result = inspect_local(codex_path=tmp_path / 'missing-codex', docker_path=tmp_path / 'missing-docker')
    assert result['observed']['codex_candidate'] == 'not_found_at_candidate_path'
    assert result['observed']['docker_candidate'] == 'not_found_at_candidate_path'
    assert not result['ready_to_run']
