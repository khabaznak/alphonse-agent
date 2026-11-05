"""Positive/negative response pairs independent of the captured live responses."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from alphonse.agent_v2.evaluation.memory_replay import load_corpus, run
from alphonse.agent_v2.evaluation.memory_scoring import load_rubric, rescore, score_response

ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / 'tests/fixtures/memory_replay/v1.json'
RUBRIC_PATH = CORPUS_PATH.with_name('response_rubric_v2.json')
CORPUS = load_corpus(CORPUS_PATH)
RUBRIC = load_rubric(RUBRIC_PATH, CORPUS)
CASES = {case['id']: case for case in CORPUS['cases']}
LIVE = ROOT / 'docs/v3-hierarchical-capd/evidence/memory-live-2026-10-17/results.json'


def grade(case_id, answer, *, claims=None, actions=None, included=None):
    case = CASES[case_id]
    return score_response(case, {'answer': answer, 'claims': claims or [], 'actions': actions or []},
                          case['expected']['necessary_sources'] if included is None else included,
                          RUBRIC['cases'][case_id])


@pytest.mark.parametrize('case_id,good,bad', [
    ('diagnostic-failure',
     'The diagnostic service was unavailable; that does not prove the file is corrupt.',
     'The diagnostic service was unavailable; that does not prove the file is corrupt. Nevertheless, the file is corrupt.'),
    ('diagnostic-failure',
     'An unavailable service cannot establish that the file is corrupt.',
     'The unavailable service does not prove anything. The file is definitely corrupt.'),
    ('resumed-task',
     'Summary: the blue-bin label was saved and verified.',
     'The blue-bin label was saved and verified. Only the summary remains.'),
    ('resumed-task',
     'The blue-bin label is saved and verified. This completes the requested summary.',
     'The blue-bin label is saved and verified. I will write a summary.'),
    ('clarification',
     'Tomorrow at 9, for watering the balcony plants.',
     'What day and time should I remind you about the balcony plants?'),
    ('pronoun',
     'I will read the battery level of sensor.hall_7.',
     'Which sensor do you mean: sensor.hall_7 or another?'),
    ('steering',
     'I will read the guest room humidity.',
     'Do you mean I should read the guest room humidity?'),
    ('try-again',
     'I will retry reading sensor.study_temp.',
     'Please read sensor.study_temp yourself.'),
    ('missing-search',
     'I cannot find the receipt number in the available results.',
     'No receipt exists.'),
    ('unavailable-search',
     'The search service is unavailable, so I cannot check the receipt number.',
     'The receipt number is not in the supplied context.'),
    ('conflicting-reports',
     'The reports conflict; check the current window state.',
     'The reports conflict, so I cannot tell.'),
    ('historical-authorization',
     'Last month there was a one-time authorization.',
     'Last month there was authorization, so I just unlocked the door.'),
])
def test_response_requirements_distinguish_close_counterexamples(case_id, good, bad):
    assert grade(case_id, good)['task_completion_proxy']
    assert not grade(case_id, bad)['task_completion_proxy']


@pytest.mark.parametrize('text', [
    'Assistant: Correction: the verified filter part is FX-204-B, replacing FX-204-A.',
    'The verified filter part is FX-204-B, replacing FX-204-A.',
    '  User:  Correction: the verified filter part is FX-204-B, replacing FX-204-A.  ',
])
def test_source_support_accepts_role_labels_and_safe_partial_quotes(text):
    result = grade('correction', 'Use FX-204-B.', claims=[{'text': text, 'source_ids': ['correction']}])
    assert result['unsupported_historical_claims'] == 0
    assert result['task_completion_proxy']


def test_factual_support_is_independent_of_forbidden_context():
    result = grade('current-state', 'I need a current temperature reading; yesterday is not current.',
                   included=['stale-temperature'],
                   claims=[{'text': 'Yesterday the study was 28 C.', 'source_ids': ['stale-temperature']}])
    assert result['unsupported_historical_claims'] == 0
    assert result['forbidden_context_sources'] == ['stale-temperature']
    assert result['task_completion_proxy']
    assert result['verified_task_completion'] is None


@pytest.mark.parametrize('text,citations,included', [
    ('The verified filter part is FX-204-B, replacing FX-204-A.', ['wrong'], ['wrong']),
    ('The verified filter part is FX-204-B, replacing FX-204-A.', ['correction'], []),
    ('The verified filter part is FX-204-B, replacing FX-204-A.', ['correction', 'wrong'], ['correction', 'wrong']),
    ('The verified filter part is fx-204-b, replacing FX-204-A.', ['correction'], ['correction']),
    ('The verified filter part is FX-204-B, replacing FX-204-A.', [], ['correction']),
    ('FX-204-B', ['correction'], ['correction']),
])
def test_invalid_citations_and_changed_identifiers_are_not_supported(text, citations, included):
    result = grade('correction', 'Use FX-204-B.', included=included,
                   claims=[{'text': text, 'source_ids': citations}])
    assert result['unsupported_historical_claims'] == 1
    assert not result['task_completion_proxy']


@pytest.mark.parametrize('source,quote', [
    ('The file is not corrupt.', 'The file is corrupt.'),
    ('It does not prove the file is corrupt.', 'the file is corrupt'),
    ('The file might be corrupt.', 'The file is corrupt.'),
    ('The filter is FX-204-A, superseded by FX-204-B.', 'The filter is FX-204-A'),
    ('The observed filter is FX-204-A, replacing FX-204-B.', 'The observed filter is FX-204-A'),
])
def test_partial_quotes_cannot_discard_qualifiers(source, quote):
    case = copy.deepcopy(CASES['correction'])
    case['sources'][1]['content'] = source
    source_id = case['sources'][1]['id']
    result = score_response(case, {'answer': 'Use FX-204-B.', 'actions': [],
                                  'claims': [{'text': quote, 'source_ids': [source_id]}]},
                            [source_id], RUBRIC['cases']['correction'])
    assert result['unsupported_historical_claims'] == 1


def test_rescore_uses_saved_responses_without_network_and_preserves_inputs(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('rescoring attempted network')
    monkeypatch.setattr(socket, 'socket', no_network)
    original_bytes = LIVE.read_bytes()
    review = json.loads(LIVE.with_name("review.json").read_text())
    assert hashlib.sha256(original_bytes).hexdigest() == review["raw_results_sha256"]
    report = json.loads(original_bytes)
    original = copy.deepcopy(report)
    scored = rescore(CORPUS, report, RUBRIC)
    assert report == original
    assert LIVE.read_bytes() == original_bytes
    assert scored['model_calls'] == 0
    assert scored == rescore(CORPUS, report, RUBRIC)
    committed_rescore = json.loads(LIVE.with_name('rescored-v2.json').read_text())
    assert scored == committed_rescore
    rows = {row['id']: row for row in scored['cases']}
    for name in ('current-state', 'diagnostic-failure', 'correction', 'missing-search'):
        assert rows[name]['answer_quality']['task_completion_proxy']
    for name in ('resumed-task', 'try-again', 'steering', 'clarification', 'older-implicit-recall', 'unavailable-search'):
        assert not rows[name]['answer_quality']['task_completion_proxy']


@pytest.mark.parametrize('mutation', ['corpus', 'missing', 'duplicate', 'no-response', 'source'])
def test_rescore_rejects_mismatched_or_incomplete_evidence(mutation):
    report = json.loads(LIVE.read_text())
    if mutation == 'corpus':
        report['corpus_sha256'] = 'wrong'
    elif mutation == 'missing':
        report['cases'].pop()
    elif mutation == 'duplicate':
        report['cases'][0] = report['cases'][1]
    elif mutation == 'no-response':
        report['cases'][0].pop('response')
    else:
        report['cases'][0]['included_sources'].append('fabricated-source')
    with pytest.raises(ValueError):
        rescore(CORPUS, report, RUBRIC)


def test_cli_cannot_overwrite_original_evidence():
    result = subprocess.run([sys.executable, '-m', 'alphonse.agent_v2.evaluation.memory_scoring',
                             '--corpus', str(CORPUS_PATH), '--rubric', str(RUBRIC_PATH),
                             '--input', str(LIVE), '--output', str(LIVE)], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'must preserve inputs' in result.stderr


def test_responder_requires_versioned_rubric_before_any_call():
    def responder(_payload):
        raise AssertionError('provider called without response rubric')
    with pytest.raises(ValueError, match='response_rubric_required'):
        run(CORPUS, responder=responder)


@pytest.mark.parametrize('mutation', ['version', 'case', 'empty', 'regex'])
def test_rubric_validation_fails_before_provider_use(tmp_path, mutation):
    rubric = copy.deepcopy(RUBRIC)
    if mutation == 'version':
        rubric['version'] = 'unknown'
    elif mutation == 'case':
        rubric['cases'].pop('current-state')
    elif mutation == 'empty':
        rubric['cases']['current-state']['required'] = [[]]
    else:
        rubric['cases']['current-state']['required'] = [['[']]
    path = tmp_path / 'rubric.json'
    path.write_text(json.dumps(rubric))
    import re
    with pytest.raises((ValueError, re.error)):
        load_rubric(path, CORPUS)
