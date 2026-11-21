"""Verify the smoke obstacle's grading and real owner-worker route."""
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.evaluation.owner_gym_smoke import ScriptedProvider, grade, grade_clock, grade_dojo_clock, provider_failure_category, run


@pytest.mark.parametrize(('answer', 'passed'), [('42', True), ('41', False)])
def test_owner_worker_obstacle_scores_observed_answer(answer, passed):
    result = run(ScriptedProvider(answer))
    assert result['status'] == 'ready'
    assert result['passed'] is passed
    assert result['checks']['exact_answer'] is passed
    assert result['checks']['client_acknowledged']
    assert result['checks']['history_recorded']
    assert result['provider_calls'] == 1


def test_no_draft_is_unknown_and_incorrect_history_fails():
    assert grade('pending', None, False, [])['passed'] is None
    assert grade('failed', None, False, [])['passed'] is False
    result = grade('ready', {'kind': 'answer', 'message': '42'}, True,
                   [SimpleNamespace(content='different')])
    assert result['passed'] is False
    assert result['checks']['history_recorded'] is False


def test_provider_diagnostics_do_not_copy_error_details():
    assert provider_failure_category(ValueError('openai_codex_exec_failed: secret')) == 'openai_codex_exec_failed'
    assert provider_failure_category(ValueError('private detail')) == 'provider_error'


def test_clock_oracle_requires_observed_clock_and_correct_answer():
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value = instant.isoformat()
    clock = {'observed_at_utc': value, 'local_time': value, 'timezone': 'UTC'}
    bindings = {'read': {'tool_id': 'native.owner_clock', 'clock_observation': clock}}
    message = f'{value}\n\nHost clock observed as of: {value} (UTC; UTC {value})'
    draft = {'kind': 'answer', 'message': message}
    history = [SimpleNamespace(content=message)]
    start, end = instant - timedelta(seconds=1), instant + timedelta(seconds=1)
    assert grade_clock('ready', draft, True, history, bindings, start, end)['passed'] is True
    assert grade_clock('ready', draft, True, history, {}, start, end)['passed'] is False
    wrong = {'kind': 'answer', 'message': message.replace(value, 'wrong', 1)}
    assert grade_clock('ready', wrong, True, [SimpleNamespace(content=wrong['message'])], bindings, start, end)['passed'] is False
    assert grade_clock('ready', draft, True, history, bindings, end, end + timedelta(seconds=1))['passed'] is False


def test_dojo_clock_does_not_require_a_specific_tool_but_checks_time():
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    answer = instant.isoformat().replace('+00:00', 'Z')
    draft = {'kind': 'answer', 'message': answer}
    history = [SimpleNamespace(content=answer)]
    start, end = instant - timedelta(seconds=1), instant + timedelta(seconds=1)
    for bindings in ({}, {'one': {'tool_id': 'native.owner_clock'}}, {'two': {'tool_id': 'native.some_other_tool'}}):
        result = grade_dojo_clock('ready', draft, True, history, bindings, start, end)
        assert result['passed'] is True
        assert result['pillars']['resilience'] == 'unassessed'
    assert grade_dojo_clock('ready', draft, True, history, {}, end + timedelta(seconds=2), end + timedelta(seconds=3))['passed'] is False
    bad = {'kind': 'answer', 'message': '2026-01-01T00:00:00+02:00'}
    assert grade_dojo_clock('ready', bad, True, [SimpleNamespace(content=bad['message'])], {}, start, end)['passed'] is False
