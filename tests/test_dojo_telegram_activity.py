from alphonse.agent_v2.evaluation.dojo_telegram_activity import assess, run


def test_activity_assessment_requires_observable_signal_before_reply():
    reply = {'kind': 'reply', 'chat_id': '123'}
    good = {'kind': 'reaction', 'chat_id': '123', 'message_id': '41'}
    wrong_message = {'kind': 'reaction', 'chat_id': '123', 'message_id': '42'}
    started, released = {'kind': 'model_started'}, {'kind': 'model_released'}
    assert assess([started, good, released, reply], model_was_held=True, completed=True)['result'] == 'pass'
    assert assess([started, released, reply, good], model_was_held=True, completed=True)['result'] == 'fail'
    assert assess([started, wrong_message, released, reply], model_was_held=True, completed=True)['result'] == 'fail'
    assert assess([started, {'kind': 'typing', 'chat_id': '123'}, released, reply], model_was_held=True, completed=True)['result'] == 'pass'
    assert assess([good, started, released, reply], model_was_held=True, completed=True)['result'] == 'fail'
    assert assess([good], model_was_held=True, completed=False)['result'] == 'inconclusive'


def test_activity_dojo_exercises_the_actual_owner_worker():
    result = run(hold_seconds=.5, activity_interval=.1)
    assert result['case'] == 'telegram-owner-activity-feedback'
    assert result['checks']['model_was_held']
    assert result['checks']['reply_completed']
    assert result['checks']['result'] == 'pass'
    events = result['events']
    receipts = [event for event in events if event['kind'] == 'reaction']
    assert receipts == [{'kind': 'reaction', 'chat_id': '123', 'message_id': '41', 'emoji': '👀'}]
    assert events.index(receipts[0]) < next(i for i, event in enumerate(events) if event['kind'] == 'reply')
    assert sum(event['kind'] == 'typing' for event in events) >= 2


def test_activity_refresh_and_feedback_failure_do_not_replay_the_task():
    result = run(hold_seconds=.6, feedback_failure=True, activity_interval=.1)
    events = result['events']
    assert result['checks']['result'] == 'pass'
    kinds = [event['kind'] for event in events]
    assert kinds.count('feedback_failed') == 1
    assert kinds.count('model_started') == 1
    assert kinds.count('reply') == 1
    assert kinds.count('typing') >= 2
    assert kinds.index('feedback_failed') < kinds.index('reply')
    assert kinds.index('typing') < kinds.index('model_released') < kinds.index('reply')


def test_failed_task_stops_activity_without_a_success_reply():
    result = run(hold_seconds=.3, activity_interval=.1, model_failure=True)
    kinds = [event['kind'] for event in result['events']]
    assert kinds.count('model_started') == 1
    assert 'task_failed' in kinds
    assert 'reply' not in kinds
    assert 'typing' in kinds[:kinds.index('model_released')]
    assert 'typing' not in kinds[kinds.index('task_failed') + 1:]


def test_cancelled_task_stops_activity_while_model_is_blocked():
    result = run(hold_seconds=.3, activity_interval=.1, cancel_task=True)
    kinds = [event['kind'] for event in result['events']]
    assert kinds.count('model_started') == 1
    assert 'typing' in kinds[:kinds.index('cancellation_requested')]
    assert 'typing' not in kinds[kinds.index('cancellation_settled'):]
    assert 'reply' not in kinds


def test_question_handoff_stops_activity_and_requests_input():
    result = run(hold_seconds=.3, activity_interval=.1, needs_input=True)
    events = result['events']
    kinds = [event['kind'] for event in events]
    replies = [event for event in events if event['kind'] == 'reply']
    assert kinds.count('model_started') == 1
    assert replies == [{'kind': 'reply', 'chat_id': '123', 'text': 'Which synthetic source?'}]
    assert 'typing' in kinds[:kinds.index('model_released')]
    assert 'typing' not in kinds[kinds.index('reply') + 1:]


def test_delayed_typing_does_not_block_reply_or_replay_but_can_arrive_late():
    result = run(hold_seconds=.2, activity_interval=.1, delay_typing=True)
    kinds = [event['kind'] for event in result['events']]
    assert kinds.count('model_started') == 1
    assert kinds.count('typing_started') == 1
    assert kinds.count('reply') == 1
    assert kinds.index('typing_started') < kinds.index('model_released') < kinds.index('reply')
    assert kinds.index('reply') < kinds.index('feedback_released') < kinds.index('typing')
    assert 'typing_started' not in kinds[kinds.index('reply') + 1:]


def test_delayed_receipt_does_not_block_reply_or_replay():
    result = run(hold_seconds=.2, activity_interval=.1, delay_receipt=True)
    kinds = [event['kind'] for event in result['events']]
    assert kinds.count('model_started') == 1
    assert kinds.count('receipt_started') == 1
    assert kinds.count('reply') == 1
    assert kinds.index('receipt_started') < kinds.index('model_released') < kinds.index('reply')
    assert kinds.index('reply') < kinds.index('feedback_released') < kinds.index('reaction')
    assert 'typing' not in kinds[kinds.index('reply') + 1:]


def test_cancelled_delayed_typing_does_not_restart_after_release():
    result = run(hold_seconds=.2, activity_interval=.1, cancel_task=True, delay_typing=True)
    kinds = [event['kind'] for event in result['events']]
    assert kinds.count('model_started') == 1
    assert kinds.count('typing_started') == 1
    assert 'reply' not in kinds
    assert kinds.index('typing_started') < kinds.index('cancellation_requested')
    assert kinds.index('cancellation_settled') < kinds.index('feedback_released') < kinds.index('typing')
    assert 'typing_started' not in kinds[kinds.index('feedback_released') + 1:]
