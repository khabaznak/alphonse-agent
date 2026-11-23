import json
from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation.dojo_action_trace import summarize_six_facilities_journal, summarize_synthetic_trace
from alphonse.agent_v2.evaluation.owner_dojo_shopping import _provider_event_types


def test_provider_event_projection_discards_commands_and_private_payloads():
    payload = json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution',
        'command': 'private fixture command', 'aggregated_output': 'private fixture output'}})
    assert _provider_event_types(payload) == [{'event': 'provider_event_type',
        'event_type': 'item.completed', 'item_type': 'command_execution'}]
    with pytest.raises(ValueError):
        _provider_event_types(json.dumps({'type': 'private data', 'item': {}}))


def test_provider_events_are_not_interpreted_as_complete_history():
    report = summarize_synthetic_trace([{'event': 'provider_event_type',
        'event_type': 'item.completed', 'item_type': 'web_search'}], {})
    assert report['provider_event_types'] == [{'event_type': 'item.completed', 'item_type': 'web_search'}]
    assert report['provider_action_history'] == 'incomplete'


def test_historical_false_search_observation_remains_reviewable():
    evidence = Path(__file__).resolve().parents[1] / 'docs/agent-loop/evidence/dojo-six-facilities-20261122-04.jsonl'
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    report = summarize_six_facilities_journal(evidence)
    assert report['answer'] == rows[-1]['answer']
    assert report['native_tool_selections'] == []
    assert report['fixture_service_requests'] == []
    assert report['fixture_service_observation'] == 'available'
    assert report['provider_action_history'] == 'incomplete'
    assert report['claim_assessment'] == 'manual_review_required'


def test_records_selection_and_independently_observed_search_separately():
    events = [
        {'event': 'model_decision', 'kind': 'tool', 'tool_id': 'native.web_search',
         'arguments': {'query': 'Asterhaven'}},
        {'event': 'model_decision', 'kind': 'tool', 'tool_id': 'native.web_search',
         'arguments': {'query': 'Birchford'}},
    ]
    requests = [{'path': '/search', 'query': 'Asterhaven', 'result_titles': ['Asterhaven current temperature']}]
    report = summarize_synthetic_trace(events, {'answer': 'One report found.',
        'observed_readback': {'status': 'observed', 'requests': requests}})
    assert len(report['native_tool_selections']) == 2
    assert report['fixture_service_requests'] == requests
    assert report['provider_action_history'] == 'incomplete'
    assert report['claim_assessment'] == 'manual_review_required'
    events[0]['arguments']['query'] = 'changed'
    requests[0]['query'] = 'changed'
    assert report['native_tool_selections'][0]['arguments']['query'] == 'Asterhaven'
    assert report['fixture_service_requests'][0]['query'] == 'Asterhaven'


def test_missing_fixture_readback_is_unavailable_not_empty():
    report = summarize_synthetic_trace([], {'answer': 'I could not check.'})
    assert report['fixture_service_requests'] is None
    assert report['fixture_service_observation'] == 'unavailable'
    assert report['provider_action_history'] == 'incomplete'


def test_incomplete_bash_does_not_invent_an_outcome():
    report = summarize_synthetic_trace([{'event': 'bash_started', 'command': 'cat fixture'}], {})
    assert report['bash_starts'] == [{'command': 'cat fixture'}]
    assert report['bash_finishes'] == []
    assert report['claim_assessment'] == 'manual_review_required'


@pytest.mark.parametrize('events,result', [(None, {}), ([None], {}), ([], None)])
def test_invalid_inputs_rejected(events, result):
    with pytest.raises(ValueError, match='dojo_trace_invalid'):
        summarize_synthetic_trace(events, result)


def test_unfinished_or_non_synthetic_journal_is_rejected(tmp_path):
    path = tmp_path / 'journal.jsonl'
    path.write_text(json.dumps({'event': 'started', 'case': 'six-facility-current-reports-v2',
                                'scope': 'synthetic owner and search service; isolated container Bash'}) + '\n')
    with pytest.raises(ValueError, match='journal_unavailable'):
        summarize_six_facilities_journal(path)
    path.write_text(json.dumps({'event': 'started', 'case': 'other-case', 'scope': 'real data'}) + '\n'
                    + json.dumps({'event': 'finished', 'case': 'other-case'}) + '\n')
    with pytest.raises(ValueError, match='journal_unavailable'):
        summarize_six_facilities_journal(path)
