from alphonse.agent_v2.evaluation.dojo_failed_reference import run


def test_exact_failed_reference_and_rejection_plumbing():
    report = run(timeout=10)
    assert report['first_status'] == {'status': 'failed', 'failure_code': 'model_timeout'}
    assert report['second_status'] == {'status': 'failed', 'failure_code': 'model_unavailable'}
    assert report['followup_status']['status'] == 'ready'
    assert report['matches_selected_first'] and report['matches_first_timeout'] and report['not_other_failure']
    assert report['projection']['effects'] == 'not established by this diagnostic'
    assert report['foreign_rejection'] == report['stale_rejection'] == 'ValueError: owner_failure_reference_unavailable'
    assert report['inbound_counts'] == {'before_rejections': 3, 'after_foreign': 3, 'after_stale': 3}
    assert report['model_call_count'] == 3
    assert report['original_status_after_followup'] == report['first_status']
    assert report['semantic_answer_verdict'] == 'not_assessed'
