"""Nonmedical source-selection case with independent post-run scoring."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_appointment as case
from alphonse.agent_v2.evaluation.prescription_http_processors import LocalSource


def pilot(case_id, *, missing=None, altered=None):
    records = case.sources(case_id)
    if altered is not None:
        records['conversation'] = altered
    events = [{'event': 'read', 'source': name, 'value': value}
              for name, value in records.items() if name != missing]
    return SimpleNamespace(_finalized=True, audit=lambda: events)


def test_current_reference_changes_time_and_address_with_same_calendar_and_contacts():
    first, second = (case.sources(name) for name in case.CASE_IDS)
    assert first['calendar'] == second['calendar']
    assert first['contacts'] == second['contacts']
    assert first['conversation'] != second['conversation']
    for name, answer in [('plumber', {'time': '2026-04-20 10:00', 'address': '28 Elm Street'}),
                         ('electrician', {'time': '2026-04-21 14:30', 'address': '49 Pine Street'})]:
        result = case.evaluate(pilot(name), case_id=name, response=json.dumps(answer), status='completed')
        assert result['answer_matches_key'] and result['required_sources_covered']
        assert result['household_note_read'] and result['delivery'] == 'unassessed'
    wrong = case.evaluate(pilot('electrician'), case_id='electrician',
                          response=json.dumps({'time': '2026-04-20 10:00', 'address': '28 Elm Street'}), status='completed')
    assert not wrong['answer_matches_key']


@pytest.mark.parametrize('missing', ['conversation', 'calendar', 'contacts'])
def test_a_correct_answer_does_not_replace_observed_sources(missing):
    result = case.evaluate(pilot('plumber', missing=missing), case_id='plumber',
        response=json.dumps({'time': '2026-04-20 10:00', 'address': '28 Elm Street'}), status='completed')
    assert result['answer_matches_key'] and not result['required_sources_covered']


def test_changed_source_and_unfinalized_service_fail():
    wrong = case.sources('electrician')['conversation']
    result = case.evaluate(pilot('plumber', altered=wrong), case_id='plumber', response='{}', status='completed')
    assert result['source_mismatch'] and not result['required_sources_covered']
    with pytest.raises(RuntimeError, match='finalized'):
        case.evaluate(SimpleNamespace(_finalized=False), case_id='plumber', response='', status='blocked')


def test_new_routes_are_explicit_not_default_capabilities():
    base = 'http://127.0.0.1:1234/' + 'a' * 43 + '/'
    default = LocalSource(base)
    with pytest.raises(PermissionError, match='denied'):
        default.read({'source': 'calendar'})
    assert LocalSource(base, allowed_source_names=case.ALLOWED_SOURCES)


def test_candidate_can_use_real_loopback_routes_with_scripted_provider(monkeypatch):
    from alphonse.agent_v2.core.inference.models import InferenceResult
    from alphonse.agent_v2.evaluation.codex_clock_comparison import _run_simple
    from alphonse.agent_v2.evaluation.codex_subscription_provider import CodexSubscriptionProvider
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'conversation'}, 'text': None},
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'calendar'}, 'text': None},
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'contacts'}, 'text': None},
        *[{'kind': 'answer', 'tool_id': None, 'arguments': None,
           'text': json.dumps({'time': '2026-04-20 10:00', 'address': '28 Elm Street'})}] * 2,
    ])
    def fake_generate(_self, _request):
        value = next(decisions)
        return InferenceResult(json_value=value, raw_response=json.dumps(value))
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_generate)
    provider = CodexSubscriptionProvider(model='fixture')
    result = case.run('plumber', lambda url: _run_simple(url=url, provider=provider, model='fixture',
        emit=lambda _: None, cancelled=lambda: False, goal=case.GOAL, source_description=case.SOURCE_DESCRIPTION,
        allowed_source_names=case.ALLOWED_SOURCES, required_source_names=None,
        verification_mode='matched_provenance', max_source_reads=5))
    assert result['candidate_status'] == 'completed'
    assert result['answer_matches_key'] and result['required_sources_covered']
    assert not result['household_note_read']
