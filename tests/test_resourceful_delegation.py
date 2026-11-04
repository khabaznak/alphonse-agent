"""Independent evidence and failure-retention checks for the #158 evaluator."""
from contextlib import contextmanager
from copy import deepcopy
import json


from alphonse.agent_v2.evaluation import resourceful_delegation as evaluation


CORPUS = json.loads(evaluation.CORPUS.read_text())


def trial(tmp_path, kind):
    case = deepcopy(next(c for c in CORPUS['cases'] if c['kind'] == kind))
    return evaluation.Trial(tmp_path, case, 'http://127.0.0.1:1', [])


def test_versioned_repeated_corpus_runs_real_processor():
    result = evaluation.run_suite()
    failures = [(r['id'], r.get('error'), r.get('checks')) for r in result['results'] if not r['scenario_pass']]
    assert result['qualified_synthetic_runtime'], failures
    assert len(result['results']) == 28
    assert {r['repeat'] for r in result['results']} == {1, 2}
    assert {r['split'] for r in result['results']} == {'development', 'heldout'}
    assert not result['live_model_qualified'] and not result['live_site_qualified']


def test_completion_label_and_plausible_response_cannot_replace_submission(tmp_path):
    fixture = trial(tmp_path, 'form')
    fixture.task.status = 'completed'
    fixture.task.metadata['prepared_user_response'] = {'message': fixture.response()}
    result = fixture.score()
    assert not result['scenario_pass']
    assert not result['metrics']['verified_task_completion']
    fixture.receipts.append({'method': 'POST', 'status': 422, 'values': {'wrong': ['Synthetic Alex']}})
    assert not fixture.score()['scenario_pass']
    fixture.receipts[0]['status'] = 201
    assert fixture.score()['scenario_pass']
    fixture.receipts.append(deepcopy(fixture.receipts[0]))
    assert not fixture.score()['scenario_pass']  # duplicate effects never qualify


def test_proposed_schedule_cannot_qualify_as_persisted_activation(tmp_path):
    fixture = trial(tmp_path, 'schedule')
    fixture.task.status = 'completed'
    fixture.task.metadata['prepared_user_response'] = {'message': fixture.response()}
    assert not fixture.score()['scenario_pass']
    fixture.task.metadata['prepared_user_response']['message'] = 'Here is a proposal.'
    assert not fixture.score()['scenario_pass']


def test_failed_lookup_cannot_be_scored_as_verified_empty(tmp_path):
    fixture = trial(tmp_path, 'empty')
    fixture.task.status = 'completed'
    fixture.task.metadata['prepared_user_response'] = {'message': fixture.response()}
    fixture.receipts.append({'method': 'GET', 'status': 503})
    assert not fixture.score()['scenario_pass']
    fixture.receipts[0]['status'] = 200
    assert not fixture.score()['scenario_pass']  # transport success alone is insufficient
    fixture.receipts[0]['places'] = []
    assert fixture.score()['scenario_pass']


def test_every_crash_is_retained_in_each_repeat_and_fails_qualification(tmp_path, monkeypatch):
    @contextmanager
    def unused_server(_kind):
        yield 'http://127.0.0.1:1', []

    def crash(_self):
        raise RuntimeError('synthetic trial crash')

    monkeypatch.setattr(evaluation, 'fixture_server', unused_server)
    monkeypatch.setattr(evaluation.Trial, 'run', crash)
    corpus = deepcopy(CORPUS)
    corpus['cases'] = [corpus['cases'][0]]
    path = tmp_path / 'crash-corpus.json'
    path.write_text(json.dumps(corpus))
    result = evaluation.run_suite(path)
    assert len(result['results']) == 2
    assert all(r['error'] == 'RuntimeError: synthetic trial crash' for r in result['results'])
    assert not result['qualified_synthetic_runtime']
    assert not result['gates']['all_trials_recorded']


def test_outcome_prompts_do_not_supply_technical_solution():
    for case in CORPUS['cases']:
        assert not any(name in case['request'].lower() for name in ('playwright', 'selenium', 'urllib', 'python', 'bash'))
