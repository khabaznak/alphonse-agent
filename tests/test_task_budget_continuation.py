from copy import deepcopy

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetExhausted, BudgetLimits, TaskBudget


def vector(actions=0, model_calls=0, input_tokens=0, output_tokens=0):
    return dict(actions=actions, model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens)


def make_budget():
    return TaskBudget(BudgetLimits(vector(2, 2, 100, 100), vector(0, 1, 10, 10), 100, 10), started_at_ms=1000)


def resume(budget, **overrides):
    options = dict(grant_id='grant-1', pause_started_ms=1020, resumed_ms=1120,
                   increment=vector(1, 1, 30, 30), increment_caps=vector(1, 1, 30, 30),
                   elapsed_increment_ms=30, elapsed_increment_cap_ms=30)
    options.update(overrides)
    budget.apply_continuation_window(**options)


def test_window_preserves_history_and_excludes_only_attested_wait():
    budget = make_budget()
    budget.reserve('first', kind='tool', amounts=vector(actions=1), now_ms=1010)
    budget.settle('first', outcome='success')
    before = deepcopy(budget.to_dict()['reservations'])
    resume(budget)
    restored = TaskBudget.from_dict(budget.to_dict())
    assert restored.to_dict()['reservations'] == before
    assert restored.to_dict()['excluded_pause_ms'] == 100
    assert restored.last_observed_at_ms == 1120
    restored.reserve('second', kind='tool', amounts=vector(actions=1), now_ms=1150)
    assert restored.usage()['charged']['actions'] == 2
    with pytest.raises(ValueError, match='grant_used'):
        resume(restored, pause_started_ms=1150, resumed_ms=1150)


def test_running_time_is_not_excluded_and_clock_rollback_is_rejected():
    budget = make_budget()
    resume(budget)
    with pytest.raises(ValueError, match='clock_before_start'):
        budget.reserve('rollback', kind='tool', amounts=vector(actions=1), now_ms=1119)
    with pytest.raises(BudgetExhausted, match='elapsed'):
        budget.reserve('late', kind='tool', amounts=vector(actions=1), now_ms=1220)


def test_inflight_and_unknown_usage_cannot_be_extended():
    budget = make_budget()
    budget.reserve('pending', kind='model', amounts=vector(model_calls=1, input_tokens=10, output_tokens=10), now_ms=1010)
    with pytest.raises(ValueError, match='usage_unresolved'):
        resume(budget)
    budget.settle('pending', outcome='success', input_tokens=None, output_tokens=5)
    with pytest.raises(ValueError, match='usage_unresolved'):
        resume(budget)


def test_continuation_never_reopens_final_lane():
    budget = make_budget()
    budget.reserve('final', kind='model', lane='final', amounts=vector(model_calls=1, input_tokens=10, output_tokens=10), now_ms=1010)
    budget.settle('final', outcome='success', input_tokens=1, output_tokens=1)
    with pytest.raises(ValueError, match='final_started'):
        resume(budget)


def test_increment_bounds_and_overflow_leave_budget_untouched():
    budget = make_budget()
    original = budget.to_dict()
    with pytest.raises(ValueError, match='continuation_bounds'):
        resume(budget, increment=vector(actions=2))
    assert budget.to_dict() == original
    with pytest.raises(ValueError, match='continuation_bounds'):
        resume(budget, increment=vector(actions=(1 << 63)), increment_caps=vector(actions=(1 << 63)))
    assert budget.to_dict() == original


def test_tampered_pause_duration_and_duplicate_grant_are_rejected():
    budget = make_budget()
    resume(budget)
    snapshot = budget.to_dict()
    snapshot['excluded_pause_ms'] += 1
    with pytest.raises(ValueError, match='continuation_window'):
        TaskBudget.from_dict(snapshot)


def test_multiple_windows_cannot_overlap_and_round_trip():
    budget = make_budget()
    resume(budget)
    resume(budget, grant_id='grant-2', pause_started_ms=1130, resumed_ms=1200)
    restored = TaskBudget.from_dict(budget.to_dict())
    assert restored.to_dict()['excluded_pause_ms'] == 170
    with pytest.raises(ValueError, match='clock_before_start'):
        resume(restored, grant_id='grant-3', pause_started_ms=1199, resumed_ms=1201)
    snapshot = budget.to_dict()
    snapshot['continuation_windows'].append(deepcopy(snapshot['continuation_windows'][0]))
    with pytest.raises(ValueError, match='continuation_window'):
        TaskBudget.from_dict(snapshot)
