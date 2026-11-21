from alphonse.agent_v2.evaluation.owner_dojo_interruption import grade_attempt


def test_claim_loss_requires_observed_cancellation_and_pending_action():
    outcome = {'marker_seen': True, 'updated': 1, 'events': [{'status': 'cancelled'}],
               'cleanup_verified': True, 'status': 'failed', 'provider_calls': 1,
               'inflight': 'action', 'pending': {'reservation_id': 'action', 'tool_id': 'native.bash'},
               'message_id': 'fixture', 'recovery_items': [{'message_id': 'fixture',
                   'status': 'action_unconfirmed', 'can_abandon': False}]}
    assert grade_attempt(**outcome)['passed'] is True
    for update in ({'events': []}, {'events': [{'status': 'error'}]},
                   {'cleanup_verified': False}, {'status': 'pending'}, {'provider_calls': 2},
                   {'inflight': ''}, {'pending': {'reservation_id': 'different', 'tool_id': 'native.bash'}},
                   {'recovery_items': []}, {'recovery_items': [{'message_id': 'fixture',
                     'status': 'action_unconfirmed', 'can_abandon': True}]}):
        assert grade_attempt(**{**outcome, **update})['passed'] is False
