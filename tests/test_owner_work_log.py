"""The public activity projection never exposes raw tool content or effects."""
import json
import pytest

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
