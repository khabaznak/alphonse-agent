"""Contract checks for the reported request, using a disposable fixture only."""
from pathlib import Path

from alphonse.agent_v2.evaluation.owner_obstacle_gym import FixtureUnavailable, run_corpus


CORPUS = Path(__file__).parent / 'fixtures/owner_obstacles/relative_reminder_development_v1.json'


class ScriptedReminder:
    offline = True

    def __init__(self, *, due='2026-11-18T04:32:00+00:00', count=1, read_clock=True):
        self.due = due
        self.count = count
        self.read_clock = read_clock

    def run(self, scenario, world):
        if self.read_clock:
            assert world.act(scenario['actor'], 'read_clock', {})['status'] == 'read'
        for index in range(self.count):
            world.act(scenario['actor'], 'create_reminder', {
                'request_id': f'fixture-{index}', 'message': 'bajar por agua', 'due_at': self.due,
            })
        return {'status': 'completed'}


def test_obstacle_is_unavailable_without_an_adapter():
    row = run_corpus(CORPUS)['cases'][0]
    assert row['availability'] == 'unavailable' and row['passed'] is None


def test_scripted_effect_exercises_oracle_without_claiming_autonomous_success():
    row = run_corpus(CORPUS, ScriptedReminder())['cases'][0]
    assert row['boundary_passed'] is True
    assert row['passed'] is None and row['semantic_assessment'] == 'unassessed'
    assert len(row['effects']) == 1


def test_wrong_time_duplicate_or_missing_clock_fails():
    wrong = run_corpus(CORPUS, ScriptedReminder(due='2026-11-18T04:33:00+00:00'))['cases'][0]
    duplicate = run_corpus(CORPUS, ScriptedReminder(count=2))['cases'][0]
    missing_clock = run_corpus(CORPUS, ScriptedReminder(read_clock=False))['cases'][0]
    no_reminder = run_corpus(CORPUS, ScriptedReminder(count=0))['cases'][0]
    assert wrong['passed'] is False and wrong['checks']['effects'] is False
    assert duplicate['passed'] is False and duplicate['checks']['effects'] is False
    assert missing_clock['passed'] is False and missing_clock['checks']['required_observations'] is False
    assert no_reminder['passed'] is False and no_reminder['checks']['effects'] is False


def test_unsupported_path_is_not_a_task_pass():
    class Unsupported:
        offline = True
        def run(self, scenario, world):
            raise FixtureUnavailable()
    row = run_corpus(CORPUS, Unsupported())['cases'][0]
    assert row['availability'] == 'unavailable' and row['passed'] is None and row['effects'] == []
