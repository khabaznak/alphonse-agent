from alphonse.agent_v2.evaluation.dojo_held_followups import run


def test_scripted_dojo_observes_fifo_and_context_without_claiming_semantic_pass():
    result = run(scenario='enqueue')
    assert result['observation'] == 'followup_execution'
    assert result['initial_ready'] and result['first_ready'] and result['second_ready']
    assert result['accepted_statuses'] == ['queued'] * 3
    assert result['listed_ids'] == [s * 32 for s in ('1', '2', '3')]
    assert result['remove_status'] == 'removed'
    assert result['before_ack'] == [('1' * 32, 'queued'), ('3' * 32, 'queued')]
    assert result['before_initial_ack'] == [('1' * 32, 'queued'), ('3' * 32, 'queued')]
    assert result['before_first_ack'] == [('1' * 32, 'released'), ('3' * 32, 'queued')]
    assert result['after'] == [('1' * 32, 'completed'), ('3' * 32, 'completed')]
    assert result['inbound_count'] == 0
    assert result['model_calls'] == 3
    assert result['context_visible'] == [False, False, True]
    assert result['passed'] is None
    assert result['semantic_execution_assessed'] is False


def test_uncertain_delivery_survives_restart_without_dispatch():
    result = run(scenario='uncertain')
    assert result['initial_ready']
    assert result['held'] == [('1' * 32, 'needs_attention'), ('3' * 32, 'needs_attention')]
    assert result['model_calls'] == 1
    assert result['passed'] is None


def test_failed_predecessor_holds_followups_without_dispatch():
    result = run(scenario='failure')
    assert result['initial_status'] == 'failed'
    assert result['held'] == [('1' * 32, 'needs_attention'), ('3' * 32, 'needs_attention')]
    assert result['model_calls'] == 1
    assert result['passed'] is None


def test_lost_add_response_can_be_reconciled_after_restart():
    result = run(scenario='draft_restart')
    assert result['saved_ids'] == ['1' * 32]
    assert result['resolution'] == 'submitted'
    assert result['held_ids'] == ['1' * 32]
    assert result['model_calls'] == 1
    assert result['passed'] is None


def test_checked_unsubmitted_draft_fences_later_ipc_add():
    result = run(scenario='draft_fence')
    assert result['resolution'] == 'not_submitted'
    assert result['late_add_rejected']
    assert result['held_ids'] == []
    assert result['passed'] is None


def test_concurrent_ipc_add_and_reconcile_serialize_safely():
    result = run(scenario='draft_race')
    assert (result['add_result'], result['resolution'], result['held_ids']) in [
        ('rejected', 'not_submitted', []), ('queued', 'submitted', ['1' * 32]),
    ]
    assert result['passed'] is None


def test_actual_loop_receives_steering_and_late_steer_is_rejected():
    result = run(scenario='steer')
    assert result['submission'] == 'accepted'
    assert result['initial_status'] == 'ready'
    assert result['steering_status'] == 'included_in_planning'
    assert result['context_visible'] == [False, True]
    assert result['late_status'] == 'not_accepted'
    assert result['model_calls'] == 2
    assert result['passed'] is None


def test_cancellation_holds_steer_and_followup_for_attention():
    result = run(scenario='cancel')
    assert result['steer_submitted'] == 'accepted'
    assert result['queued'] == 'queued'
    assert result['cancel_status'] == 'cancel_requested'
    assert result['initial_status'] == 'failed'
    assert result['steer_status'] == 'needs_attention'
    assert result['held'] == [('3' * 32, 'needs_attention')]
    assert result['model_calls'] == 1
    assert result['passed'] is None


def test_hypothetical_chat_queue_has_ordered_context_without_claiming_semantic_pass():
    result = run(scenario='hypothetical')
    assert result['initial_ready'] and result['first_ready'] and result['second_ready']
    assert result['before_first_ack'] == [('1' * 32, 'released'), ('3' * 32, 'queued')]
    assert result['after'] == [('1' * 32, 'completed'), ('3' * 32, 'completed')]
    assert result['context_visible'] == [False, False, True]
    assert result['passed'] is None


def test_unknown_named_exercise_is_rejected_before_starting_a_daemon():
    import pytest
    with pytest.raises(ValueError, match='unknown_dojo_scenario'):
        run(scenario='typo')


def test_named_journal_records_provider_and_does_not_overwrite(tmp_path, monkeypatch):
    import json
    import pytest
    from alphonse.agent_v2.evaluation import dojo_held_followups as exercise

    fake = lambda **kwargs: {'observation': 'followup_execution', 'passed': None, 'received': kwargs['scenario']}
    monkeypatch.setattr(exercise, 'run', fake)
    path = tmp_path / 'receipt.json'
    exercise.main(['--scenario', 'enqueue', '--journal', str(path)])
    record = json.loads(path.read_text())
    assert record['scenario'] == record['observation']['received'] == 'enqueue'
    assert record['provider'] == 'scripted'
    assert record['observation']['passed'] is None
    with pytest.raises(FileExistsError):
        exercise.main(['--scenario', 'steer', '--journal', str(path)])
    assert json.loads(path.read_text()) == record


def test_failed_named_exercise_records_unknown_and_fails(tmp_path, monkeypatch):
    import json
    import pytest
    from alphonse.agent_v2.evaluation import dojo_held_followups as exercise

    def unavailable(**kwargs):
        raise OSError('synthetic local setup failure')
    monkeypatch.setattr(exercise, 'run', unavailable)
    path = tmp_path / 'failed.json'
    with pytest.raises(SystemExit) as exit_info:
        exercise.main(['--scenario', 'steer', '--journal', str(path)])
    assert exit_info.value.code == 1
    observation = json.loads(path.read_text())['observation']
    assert observation['observation'] == 'exercise_error'
    assert observation['error_type'] == 'OSError'
    assert observation['passed'] is None
