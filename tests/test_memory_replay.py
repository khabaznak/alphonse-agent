"""Offline replay is a characterization of real assembly, not a scripted answer test."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from alphonse.agent_v2.evaluation.memory_replay import assemble, load_corpus, run, score_answer

from alphonse.agent_v2.evaluation.memory_scoring import load_rubric

CORPUS = Path(__file__).parent / 'fixtures/memory_replay/v1.json'
RUBRIC_PATH = CORPUS.with_name('response_rubric_v2.json')
RUBRIC = load_rubric(RUBRIC_PATH, load_corpus(CORPUS))
BASELINE = Path(__file__).parents[1] / 'docs/v3-hierarchical-capd/evidence/memory-baseline-v1/results.json'


def test_offline_replay_is_reproducible_without_network(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('offline replay attempted network access')
    monkeypatch.setattr(socket, 'socket', no_network)
    corpus = load_corpus(CORPUS)
    first = run(corpus)
    candidate = BASELINE.parent.parent / 'memory-knowledge-v2/results.json'
    assert first == run(corpus) == json.loads(candidate.read_text())
    assert first['adapter'] == 'current-v3-context-and-native-search-v2'
    # Preserve the frozen characterization evidence; do not overwrite it to
    # accommodate intentional runtime prompt changes.
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == '5306f97c018da232af46cd511c532306662685ac659d793683a04a08e5c4be99'
    assert all(c['answer_quality'] is None and c['model_calls'] == 0 and c['latency_ms'] is None
               for c in first['cases'])


def test_private_and_shared_records_exercise_real_owner_filter():
    corpus = load_corpus(CORPUS)
    case = next(c for c in corpus['cases'] if c['id'] == 'shared-visibility')
    observed = assemble(case)
    assert 'shared' in observed['included_sources']
    assert 'private-b' not in observed['included_sources']
    # A private record becomes visible only under the actual owner's account.
    changed = copy.deepcopy(case)
    changed['user'] = 'member-b'
    assert 'private-b' in assemble(changed)['included_sources']


def test_unavailable_search_is_distinct_from_missing_results():
    rows = {r['id']: r for r in run(load_corpus(CORPUS))['cases']}
    empty, unavailable = rows['missing-search'], rows['unavailable-search']
    assert empty['search_backend_states'] == ['empty']
    assert unavailable['search_backend_states'] == ['unavailable']
    assert empty['retrieved_sources'] == unavailable['retrieved_sources'] == []
    # Current native search swallows the error: retain this known gap in baseline.
    assert empty['search_error'] == unavailable['search_error'] == ''


def test_old_explicit_recall_uses_search_while_implicit_recall_is_missing():
    corpus = load_corpus(CORPUS)
    explicit = assemble(next(c for c in corpus['cases'] if c['id'] == 'older-explicit-recall'))
    implicit = assemble(next(c for c in corpus['cases'] if c['id'] == 'older-implicit-recall'))
    assert explicit['retrieved_sources'] == ['old-filter']
    assert 'old-shelf' not in implicit['included_sources']


def test_live_scoring_rejects_unsupported_claims_and_historical_permissions():
    case = next(c for c in load_corpus(CORPUS)['cases'] if c['id'] == 'historical-authorization')
    trace = dict(answer='Last month you authorized it.', actions=[],
                 claims=[dict(text='Last month the user authorized unlocking the door once.', source_ids=['old-permission'])])
    assert score_answer(case, trace, ['old-permission'], rules=RUBRIC['cases'][case['id']])['task_completion_proxy']
    assert score_answer(case, trace, [], rules=RUBRIC['cases'][case['id']])['unsupported_historical_claims'] == 1
    trace['actions'] = ['unlock-door']
    assert not score_answer(case, trace, ['old-permission'], rules=RUBRIC['cases'][case['id']])['task_completion_proxy']
    trace['claims'][0]['text'] = 'The door is unlocked now.'
    assert score_answer(case, trace, ['old-permission'], rules=RUBRIC['cases'][case['id']])['unsupported_historical_claims'] == 1


def test_optional_responder_gets_observed_context_without_answer_oracle():
    corpus = load_corpus(CORPUS)
    corpus['cases'] = corpus['cases'][:1]
    inputs = []
    def responder(payload):
        inputs.append(payload)
        return dict(answer='I need to read the sensor.', actions=[], claims=[])
    row = run(corpus, responder=responder, rubric=RUBRIC)['cases'][0]
    assert row['model_calls'] == 1
    assert row['answer_quality']['task_completion_proxy']
    assert row['answer_quality']['semantic_review_required']
    assert set(inputs[0]) == {'message', 'context', 'source_ids', 'response_contract'}
    assert 'SYNTHETIC-998' not in inputs[0]['context']


@pytest.mark.parametrize('args', [['--live'], ['--responder', 'does_not_exist:run']])
def test_cli_requires_both_explicit_live_flags(tmp_path, args):
    result = subprocess.run([sys.executable, '-m', 'alphonse.agent_v2.evaluation.memory_replay_live',
                             '--corpus', str(CORPUS), '--rubric', str(RUBRIC_PATH), '--output', str(tmp_path / 'result.json'), *args],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert '--live and --responder must be supplied together' in result.stderr
    assert not (tmp_path / 'result.json').exists()


def test_corpus_rejects_unknown_source_expectations(tmp_path):
    corpus = load_corpus(CORPUS)
    corpus['cases'][0]['expected']['necessary_sources'] = ['nonexistent']
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match='unknown_expected_source'):
        load_corpus(path)
