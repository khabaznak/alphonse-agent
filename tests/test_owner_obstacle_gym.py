"""Synthetic fixture contracts, not model performance or live integration tests."""
from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation.owner_obstacle_gym import ClockCandidateAdapter, FixtureInterrupted, FixtureWorld, ScriptedCandidateAdapter, load_corpus, run_corpus


CORPUS = Path(__file__).parent / 'fixtures/owner_obstacles/development_v1.json'
CLOCK_CORPUS = Path(__file__).parent / 'fixtures/owner_obstacles/clock_development_v1.json'


class ScriptedAdapter:
    offline = True
    def run(self, scenario, world):
        assert set(scenario).isdisjoint({'oracle', 'expected_effects', 'allowed_statuses'})
        goal = scenario['goal']
        actor = scenario['actor']
        if 'plumber' in goal:
            args = {'request_id': 'reminder-1', 'message': 'call the plumber', 'due_at': '2026-01-02T10:00:00+00:00'}
            assert world.act(actor, 'create_reminder', args)['status'] == 'saved'
            assert world.act(actor, 'create_reminder', args)['status'] == 'duplicate'
            return {'status': 'completed'}
        if 'skills' in goal:
            for skill in ('pantry', 'meal-plan'):
                assert world.act(actor, 'read_skill', {'name': skill})['status'] == 'read'
            assert world.act(actor, 'read_skill', {'name': 'private'})['status'] == 'denied'
            return {'status': 'completed'}
        operation, args = {
            'Schedule a weekly household meeting.': ('schedule_event', None),
            'Schedule the household meeting every Monday at 10:00 UTC, starting 2026-01-05.': ('schedule_event', {'title': 'household meeting', 'recurrence': 'weekly', 'timezone': 'UTC', 'starts_at': '2026-01-05T10:00:00+00:00'}),
            'Start a 25-minute focus timer.': ('start_timer', {'label': 'focus', 'duration_seconds': 1500}),
            'When the synthetic door opens, remind me to check the lights.': ('subscribe_event', {'source': 'synthetic door', 'event': 'open', 'action': 'remind check lights'}),
            'Reserve a 30-minute focus block after lunch.': ('schedule_timebox', None),
            'Reserve a 30-minute focus block on 2026-01-01 at 13:00 UTC.': ('schedule_timebox', {'label': 'focus', 'start': '2026-01-01T13:00:00+00:00', 'duration_minutes': 30}),
            'Read the answer aloud.': ('play_voice', None),
            'Read aloud exactly: synthetic answer.': ('play_voice', {'text': 'synthetic answer'}),
            'Turn off the synthetic kitchen light.': ('home_assistant_write', {'entity_id': 'light.synthetic_kitchen', 'state': 'off'}),
        }[goal]
        observed = world.act(actor, operation, {'request_id': 'fixture', **args} if args is not None else {})['status']
        return {'status': 'completed' if observed == 'saved' else 'blocked' if observed == 'denied' else 'unavailable'}


def test_default_preserves_every_case_as_unavailable():
    report = run_corpus(CORPUS)
    assert len(report['cases']) == 15
    assert all(row['availability'] == 'unavailable' and row['passed'] is None for row in report['cases'])
    assert report['simulation'] is True
    assert any('process interruption may lose all in-memory rows' in text for text in report['limitations'])


def test_fixture_adapter_exercises_boundaries_without_claiming_unsupported_completion():
    report = run_corpus(CORPUS, ScriptedAdapter())
    rows = {row['id']: row for row in report['cases']}
    assert rows['reminder-save-once']['passed'] is True
    assert len(rows['reminder-save-once']['effects']) == 1
    assert rows['compose-two-authorized-skills']['passed'] is None
    assert rows['compose-two-authorized-skills']['boundary_passed'] is True
    assert rows['compose-two-authorized-skills']['semantic_assessment'] == 'unassessed'
    for key in ('recurring-event-not-active', 'timebox-not-started', 'event-trigger-not-installed', 'timebox-not-scheduled', 'voice-unavailable', 'home-assistant-mutation-unavailable'):
        assert rows[key]['status'] == 'unavailable'
        assert rows[key]['boundary_passed'] is True and rows[key]['passed'] is None
        assert rows[key]['target_status'] == 'completed'
        assert rows[key]['effects'] == []
    for key in ('recurring-event-simulated', 'timer-simulated', 'event-trigger-simulated', 'timebox-simulated', 'voice-simulated', 'home-assistant-simulated'):
        assert rows[key]['status'] == 'completed' and rows[key]['passed'] is True
        assert len(rows[key]['effects']) == 1
    assert rows['home-assistant-denied']['status'] == 'blocked'
    assert rows['home-assistant-denied']['passed'] is None and rows['home-assistant-denied']['effects'] == []


def test_real_candidate_fixture_adapter_exercises_skill_observations():
    rows = {row['id']: row for row in run_corpus(CORPUS, ScriptedCandidateAdapter())['cases']}
    skill = rows['compose-two-authorized-skills']
    assert skill['status'] == 'completed' and skill['passed'] is None and skill['boundary_passed'] is True
    assert skill['semantic_assessment'] == 'unassessed'
    assert [event['resource'] for event in skill['events']] == ['pantry', 'meal-plan']
    assert all(row['availability'] == 'unavailable' and row['passed'] is None
               for key, row in rows.items() if key != 'compose-two-authorized-skills')


def test_independent_oracle_rejects_fabricated_success_and_wrong_effect():
    class Fabricator:
        offline = True
        def run(self, scenario, world):
            if 'plumber' in scenario['goal']:
                world.act('owner', 'create_reminder', {'request_id': 'wrong', 'message': 'wrong', 'due_at': 'wrong'})
            else:
                world.act('owner', 'start_timer', {})
            return {'status': 'completed'}
    report = run_corpus(CORPUS, Fabricator())
    assert all(row['passed'] is False for row in report['cases'])
    assert report['cases'][0]['checks']['effects'] is False


def test_actor_boundary_and_request_id_conflict():
    world = FixtureWorld({'owner': 'owner', 'available_operations': ['create_reminder']})
    request = {'request_id': 'one', 'message': 'fixture', 'due_at': 'fixture-time'}
    assert world.act('other', 'create_reminder', request)['status'] == 'denied'
    assert world.effects == []
    assert world.act('owner', 'create_reminder', request)['status'] == 'saved'
    assert world.act('owner', 'create_reminder', {**request, 'message': 'changed'})['status'] == 'conflict'
    assert len(world.effects) == 1


def test_interruption_and_errors_preserve_observed_effects_without_passing():
    class Interrupted:
        offline = True
        def run(self, scenario, world):
            if 'plumber' in scenario['goal']:
                world.act('owner', 'create_reminder', {'request_id': 'one', 'message': 'fixture', 'due_at': 'fixture-time'})
                raise FixtureInterrupted()
            raise RuntimeError('private detail')
    report = run_corpus(CORPUS, Interrupted())
    assert report['cases'][0]['availability'] == 'interrupted'
    assert report['cases'][0]['passed'] is False and len(report['cases'][0]['effects']) == 1
    assert report['cases'][0]['observed_effects_consistent'] is False
    assert all(row['availability'] == 'error' and row['passed'] is None for row in report['cases'][1:])
    assert 'private detail' not in repr(report)


@pytest.mark.parametrize('failure', [FixtureInterrupted, RuntimeError])
def test_interrupted_expected_effect_is_unknown_but_wrong_effect_is_failure(failure):
    class Partial:
        offline = True
        def __init__(self, message): self.message = message
        def run(self, scenario, world):
            if 'plumber' in scenario['goal']:
                world.act('owner', 'create_reminder', {'request_id': 'reminder-1', 'message': self.message, 'due_at': '2026-01-02T10:00:00+00:00'})
            raise failure('synthetic')
    correct = run_corpus(CORPUS, Partial('call the plumber'))['cases'][0]
    wrong = run_corpus(CORPUS, Partial('invented reminder'))['cases'][0]
    assert correct['passed'] is None and correct['observed_effects_consistent'] is True
    assert wrong['passed'] is False and wrong['observed_effects_consistent'] is False


def test_wrong_effect_fails_even_when_adapter_reports_blocked():
    class WrongEffect:
        offline = True
        def run(self, scenario, world):
            if 'plumber' in scenario['goal']:
                world.act('owner', 'create_reminder', {'request_id': 'bad', 'message': 'wrong', 'due_at': 'wrong'})
            return {'status': 'blocked'}
    rows = run_corpus(CORPUS, WrongEffect())['cases']
    assert rows[0]['status'] == 'blocked' and rows[0]['passed'] is False
    assert rows[0]['checks']['effects'] is False


def test_corpus_digest_and_adapter_declaration_are_enforced(tmp_path):
    copy = tmp_path / 'fixture.json'
    copy.write_bytes(CORPUS.read_bytes() + b' ')
    copy.with_suffix('.sha256').write_text(CORPUS.with_suffix('.sha256').read_text())
    with pytest.raises(ValueError, match='digest_mismatch'):
        load_corpus(copy)
    with pytest.raises(ValueError, match='declare_offline'):
        run_corpus(CORPUS, object())


def test_cli_runs_candidate_adapter_and_refuses_overwrite(tmp_path):
    import json
    import subprocess
    import sys
    output = tmp_path / 'report.json'
    command = [sys.executable, '-m', 'alphonse.agent_v2.evaluation.owner_obstacle_gym', str(CORPUS), str(output), '--adapter', 'candidate-scripted']
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report['simulation'] is True and len(report['cases']) == 15
    skill = next(row for row in report['cases'] if row['id'] == 'compose-two-authorized-skills')
    assert skill['passed'] is None and skill['boundary_passed'] is True
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_real_candidate_reads_synthetic_clock_and_fails_closed_on_unavailable_or_denied():
    rows = {row['id']: row for row in run_corpus(CLOCK_CORPUS, ClockCandidateAdapter())['cases']}
    success = rows['clock-utc-observed']
    assert success['passed'] is True and success['checks']['fixture_exact_response'] is True
    assert success['events'] == [{'actor': 'owner', 'operation': 'read_clock', 'status': 'read'}]
    assert success['effects'] == []
    for key, reason in [('clock-unavailable', 'unavailable'), ('clock-denied', 'denied')]:
        row = rows[key]
        assert row['status'] == 'blocked' and row['passed'] is None and row['boundary_passed'] is True
        assert row['events'][0]['status'] == reason and row['effects'] == []


def test_clock_oracle_rejects_fabricated_timestamp_and_missing_observation():
    class WrongClock:
        offline = True
        def run(self, scenario, world):
            world.act('owner', 'read_clock', {})
            return {'status': 'completed', 'response': '{"observed_at_utc":"1900-01-01T00:00:00+00:00"}'}
    row = run_corpus(CLOCK_CORPUS, WrongClock())['cases'][0]
    assert row['passed'] is False and row['checks']['fixture_exact_response'] is False
    class SkippedRead:
        offline = True
        def run(self, scenario, world):
            return {'status': 'completed', 'response': '{"observed_at_utc":"2026-11-17T15:04:00+00:00"}'}
    row = run_corpus(CLOCK_CORPUS, SkippedRead())['cases'][0]
    assert row['passed'] is False and row['checks']['required_observations'] is False


def test_clock_world_refuses_other_actor_and_malformed_fixture():
    clock = {'observed_at_utc': '2026-11-17T15:04:00+00:00', 'timezone': 'UTC'}
    world = FixtureWorld({'owner': 'owner', 'available_operations': ['read_clock'], 'clock': clock})
    assert world.act('other', 'read_clock', {}) == {'status': 'denied'}
    assert world.act('owner', 'read_clock', {'extra': 'x'}) == {'status': 'invalid'}
    assert world.effects == []


def test_clock_cli_runs_actual_candidate_without_model_or_provider(tmp_path):
    import json
    import subprocess
    import sys
    output = tmp_path / 'clock.json'
    command = [sys.executable, '-m', 'alphonse.agent_v2.evaluation.owner_obstacle_gym', str(CLOCK_CORPUS), str(output), '--adapter', 'clock-scripted']
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report['simulation'] is True and report['cases'][0]['passed'] is True
    assert all(row['passed'] is None for row in report['cases'][1:])
