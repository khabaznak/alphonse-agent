"""The public activity projection never exposes raw tool content or effects."""
import json
import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import observation_digest
from alphonse.agent_v2.core.intelligence.simple_loop.owner_work_log import _events


def test_sanitized_observations_preserve_order_and_unknown_effects():
    observations = {
        'one': {'actor': 'owner', 'tool_id': 'native.web_search', 'arguments': {'query': 'PRIVATE QUERY'},
                'web_observation': {'result_count': 0, 'secret': 'PRIVATE RESULT'}},
        'two': {'actor': 'owner', 'tool_id': 'native.bash', 'arguments': {'command': 'PRIVATE COMMAND'},
                'host_bash_observation': {'stdout': 'PRIVATE OUTPUT', 'effect_status': 'unverified'}},
    }
    projected = _events('owner', ['one', 'two'], observations,
                        {'kind': 'tool', 'tool_id': 'native.web_fetch', 'arguments': {'url': 'PRIVATE URL'}})
    assert projected == [
        {'label': 'Searched the web', 'outcome': 'No results returned'},
        {'label': 'Local command', 'outcome': 'Command returned; effects are unconfirmed'},
        {'label': 'Tool action', 'outcome': 'Reserved; execution or effects are unconfirmed'},
    ]
    assert 'PRIVATE' not in json.dumps(projected)


def test_foreign_observation_is_rejected():
    with pytest.raises(ValueError, match='owner_work_log_unavailable'):
        _events('owner', ['foreign'], {'foreign': {'actor': 'other', 'tool_id': 'native.search_memory'}}, None)


def test_shell_outcomes_do_not_imply_verified_effects():
    observations = {
        'timeout': {'actor': 'owner', 'tool_id': 'native.bash', 'host_bash_observation': {'timed_out': True}},
        'failure': {'actor': 'owner', 'tool_id': 'native.bash', 'host_bash_observation': {'exit_code': 1}},
    }
    events = _events('owner', ['timeout', 'failure'], observations, None)
    assert events[0]['outcome'] == 'Command timed out; effects are unconfirmed'
    assert events[1]['outcome'] == 'Command returned exit code 1; effects are unconfirmed'


def test_gmail_work_log_projects_bound_labels_for_each_search():
    def recorded(mail):
        return {'actor': 'owner', 'tool_id': 'native.owner_gmail_search',
                'arguments': {'query': 'PRIVATE QUERY', 'label': 'UNVERIFIED LABEL'},
                'gmail_search_observation': mail, 'observation_sha256': observation_digest(mail)}

    school = recorded({'status': 'partial', 'authority': {'grant_id': 'PRIVATE GRANT',
                       'resource': 'Label_42', 'label_name': 'Family/School'}, 'messages': ['PRIVATE MAIL']})
    observations = {
        'inbox': recorded({'status': 'complete', 'authority': {'grant_id': 'PRIVATE GRANT'}, 'messages': []}),
        'school': school,
    }
    refs = list(observations)
    events = _events('owner', refs, observations, None)
    assert [event['gmail_label'] for event in events] == ['Inbox', 'Family/School']
    assert all(secret not in json.dumps(events) for secret in ('PRIVATE', 'UNVERIFIED LABEL', 'Label_42'))

    observations['unavailable'] = recorded({'status': 'unavailable', 'reason': 'label_content_access_required'})
    assert 'gmail_label' not in _events('owner', ['unavailable'], observations, None)[0]
    school['observation_sha256'] = 'incorrect'
    assert 'gmail_label' not in _events('owner', ['school'], observations, None)[0]
