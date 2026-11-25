"""The public activity projection never exposes raw tool content or effects."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.owner_work_log import _activity, _events, _live_claim
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import observation_digest, _decision_digest
from alphonse.agent_v2.core.intelligence.simple_loop.candidate_decision_policy import normalize_action
from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest


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


def test_activity_counts_recorded_actions_without_exposing_or_interpreting_content():
    observations = {
        'first': {'actor': 'owner', 'tool_id': 'native.bash',
                  'arguments': {'command': 'PRIVATE COMMAND'},
                  'host_bash_observation': {'exit_code': 1, 'stdout': 'PRIVATE OUTPUT'}},
        'second': {'actor': 'owner', 'tool_id': 'native.bash',
                   'host_bash_observation': {'exit_code': 0, 'effect_status': 'verified'}},
    }
    activity = _activity('owner', ['first', 'second'], observations,
                         {'kind': 'tool', 'arguments': {'url': 'PRIVATE URL'}})
    assert activity == {
        'recorded_action_count': 2,
        'unconfirmed_effect_count': 2,
        'pending_tool': True,
        'pending_intent': None,
        'outcome_counts': [
            {'tool_kind': 'local_command', 'outcome': 'exit_nonzero', 'count': 1},
            {'tool_kind': 'local_command', 'outcome': 'exit_zero', 'count': 1},
        ],
        'latest': {'tool_kind': 'local_command', 'outcome': 'exit_zero', 'effects': 'unconfirmed'},
    }
    assert 'PRIVATE' not in json.dumps(activity)


def test_activity_pending_is_not_a_recorded_observation():
    activity = _activity('owner', [], {}, {'kind': 'tool', 'tool_id': 'native.web_search'})
    assert activity == {'recorded_action_count': 0, 'unconfirmed_effect_count': 0,
                        'pending_tool': True, 'pending_intent': None, 'outcome_counts': [], 'latest': None}


def test_activity_rejects_foreign_observation_and_does_not_treat_boolean_as_result_count():
    with pytest.raises(ValueError, match='owner_work_log_unavailable'):
        _activity('owner', ['x'], {'x': {'actor': 'other', 'tool_id': 'native.bash'}}, None)
    observation = {'actor': 'owner', 'tool_id': 'native.web_search',
                   'web_observation': {'result_count': False}}
    assert _activity('owner', ['x'], {'x': observation}, None)['latest']['outcome'] == 'recorded'


@pytest.mark.parametrize(('shell', 'expected'), [
    ({'timed_out': True, 'exit_code': 0}, 'timed_out'),
    ({'cancelled': True, 'exit_code': 0}, 'interrupted'),
    ({'exit_code': 1}, 'exit_nonzero'),
    ({'exit_code': False}, 'unknown'),
    ({'exit_code': 9999}, 'unknown'),
])
def test_activity_shell_status_is_observation_not_effect(shell, expected):
    observation = {'actor': 'owner', 'tool_id': 'native.bash', 'host_bash_observation': shell}
    latest = _activity('owner', ['x'], {'x': observation}, None)['latest']
    assert latest == {'tool_kind': 'local_command', 'outcome': expected, 'effects': 'unconfirmed'}


def test_activity_groups_prior_mixed_results_without_disclosing_details():
    def command(**result):
        return {'actor': 'owner', 'tool_id': 'native.bash',
                'arguments': {'command': 'PRIVATE COMMAND'}, 'host_bash_observation': result}
    observations = {
        'a': command(exit_code=1, stdout='PRIVATE OUTPUT'),
        'b': command(exit_code=1),
        'c': command(exit_code=0),
        'd': command(timed_out=True),
        'e': command(cancelled=True),
        'f': command(exit_code=False),
        'g': {'actor': 'owner', 'tool_id': 'native.search_memory'},
        'h': command(exit_code=0),
    }
    refs = list(observations)
    activity = _activity('owner', refs, observations, {'kind': 'tool'})
    assert activity['outcome_counts'] == [
        {'tool_kind': 'local_command', 'outcome': 'exit_nonzero', 'count': 2},
        {'tool_kind': 'local_command', 'outcome': 'exit_zero', 'count': 2},
        {'tool_kind': 'local_command', 'outcome': 'timed_out', 'count': 1},
        {'tool_kind': 'local_command', 'outcome': 'interrupted', 'count': 1},
        {'tool_kind': 'local_command', 'outcome': 'unknown', 'count': 1},
        {'tool_kind': 'conversation_search', 'outcome': 'recorded', 'count': 1},
    ]
    assert activity['latest'] == {'tool_kind': 'local_command', 'outcome': 'exit_zero',
                                  'effects': 'unconfirmed'}
    assert activity['recorded_action_count'] == 8 and activity['unconfirmed_effect_count'] == 7
    assert 'PRIVATE' not in json.dumps(activity)
    assert [event['outcome'] for event in _events('owner', refs, observations, None)][:3] == [
        'Command returned exit code 1; effects are unconfirmed',
        'Command returned exit code 1; effects are unconfirmed',
        'Command returned exit code 0; effects are unconfirmed',
    ]


@pytest.mark.parametrize(('tool', 'code', 'kind', 'detail'), [
    ('native.web_search', 'searxng_timeout', 'web_search', 'timeout'),
    ('native.web_search', 'searxng_http_error', 'web_search', 'http_error'),
    ('native.web_search', 'searxng_invalid_json', 'web_search', 'invalid_response'),
    ('native.web_fetch', 'web_fetch_timeout', 'web_source', 'timeout'),
    ('native.web_fetch', 'web_fetch_http_error', 'web_source', 'http_error'),
    ('native.web_fetch', 'web_fetch_unresolvable_host', 'web_source', 'unresolvable_host'),
    ('native.web_fetch', 'web_fetch_output_too_large', 'web_source', 'output_too_large'),
    ('native.web_fetch', 'web_fetch_output_invalid', 'web_source', 'invalid_response'),
])
def test_activity_exposes_only_bound_native_web_failure_details(tool, code, kind, detail):
    web = {'status': 'failed', 'tool_id': tool, 'error_code': code}
    item = {'actor': 'owner', 'tool_id': tool, 'web_observation': web,
            'observation_sha256': observation_digest(web), 'arguments': {'query': 'PRIVATE QUERY'}}
    latest = _activity('owner', ['x'], {'x': item}, None)['latest']
    assert latest == {'tool_kind': kind, 'outcome': 'source_failed',
                      'effects': 'not_reported', 'detail': detail}
    assert 'PRIVATE' not in json.dumps(latest)
    item['observation_sha256'] = 'incorrect'
    assert 'detail' not in _activity('owner', ['x'], {'x': item}, None)['latest']


def test_activity_does_not_publish_unrecognized_web_error():
    web = {'status': 'failed', 'tool_id': 'native.web_fetch', 'error_code': 'PRIVATE ERROR'}
    item = {'actor': 'owner', 'tool_id': 'native.web_fetch', 'web_observation': web,
            'observation_sha256': observation_digest(web)}
    assert 'detail' not in _activity('owner', ['x'], {'x': item}, None)['latest']


def test_pending_intent_requires_live_matching_actor_and_reservation():
    digest = 'a' * 64
    pending = {'kind': 'tool', 'reservation_id': 'r', 'tool_id': 'native.bash', 'claim_digest': digest,
               'arguments': {'command': 'sensitive'}}
    state = {'status': 'running', 'inflight': 'r'}
    decision = {'kind': 'tool', 'tool_id': 'native.bash', 'arguments': pending['arguments'],
                'text': None, 'activity_intent': 'check_state'}
    selected_digest = _decision_digest(decision)
    marker = {'schema_version': 1, 'actor': 'owner', 'task_id': 'task', 'reservation_id': 'r',
              'tool_id': 'native.bash', 'claim_digest': digest, 'decision_sha256': selected_digest,
              'projection_sha256': digest, 'intent': 'check_state', 'usage': None,
              'selection': {'reservation_id': 'm', 'claim_digest': digest, 'decision_sha256': selected_digest,
                            'projection_sha256': digest}}
    def project(m, s=state, p=pending):
        return _activity('owner', [], {}, p, task_id='task', state=s, intent_marker=m)['pending_intent']
    assert project(marker) == 'check_state'
    for change in ({'actor': 'other'}, {'reservation_id': 'old'}, {'claim_digest': 'b' * 64},
                   {'intent': 'PRIVATE SOURCE TEXT'}, {'intent': 'make_change'},
                   {'decision_sha256': 'b' * 64}, {'projection_sha256': 'b' * 64},
                   {'schema_version': True}, {'extra': 'PRIVATE'}):
        assert project({**marker, **change}) is None
    usage = {'input_tokens': 10, 'output_tokens': None}
    usage_digest = _decision_digest({**decision, 'usage': usage})
    with_usage = {**marker, 'usage': usage, 'decision_sha256': usage_digest,
                  'selection': {**marker['selection'], 'decision_sha256': usage_digest}}
    assert project(with_usage) == 'check_state'
    assert project({**with_usage, 'usage': {'input_tokens': 11, 'output_tokens': None}}) is None
    assert project(marker, {'status': 'blocked', 'inflight': 'r'}) is None
    assert project(marker, {'status': 'running', 'inflight': 'new'}) is None
    assert project(marker, state, None) is None


def test_action_intent_rejects_untrusted_text_and_preserves_legacy_actions():
    legacy = {'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'x'}, 'text': None}
    assert normalize_action(legacy, frozenset({'native.bash'})) == legacy
    chosen = {**legacy, 'activity_intent': 'check_state'}
    assert normalize_action(chosen, frozenset({'native.bash'})) == chosen
    assert normalize_action({**legacy, 'activity_intent': 'PRIVATE INJECTED SOURCE'},
                            frozenset({'native.bash'})) == {'kind': 'invalid'}
    assert normalize_action({'kind': 'answer', 'tool_id': None, 'arguments': None,
                             'text': 'Answer', 'activity_intent': 'make_change'}, frozenset()) == {'kind': 'invalid'}


def test_intent_requires_current_unexpired_claim_and_owner():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE v2_inbound_messages (message_id TEXT, user_id TEXT, status TEXT,
                    lease_owner TEXT, claim_generation TEXT, lease_expires_at TEXT)''')
    claim = ClaimReceipt('task', 'owner', 'worker', 'a' * 32)
    marker = {'claim_digest': _claim_digest(claim)}
    future = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    conn.execute('INSERT INTO v2_inbound_messages VALUES (?,?,?,?,?,?)',
                 ('task', 'owner', 'processing', 'worker', 'a' * 32, future))
    assert _live_claim(conn, 'task', 'owner', marker)
    assert not _live_claim(conn, 'task', 'other', marker)
    conn.execute('UPDATE v2_inbound_messages SET claim_generation=?', ('b' * 32,))
    assert not _live_claim(conn, 'task', 'owner', marker)
    conn.execute('UPDATE v2_inbound_messages SET claim_generation=?,lease_expires_at=?',
                 ('a' * 32, (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()))
    assert not _live_claim(conn, 'task', 'owner', marker)
