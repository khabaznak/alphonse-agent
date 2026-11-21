from datetime import datetime, timedelta, timezone

from alphonse.agent_v2.evaluation.dojo_reminder_followup import scenario


def test_followup_grades_effect_and_independent_simulated_offer():
    start = datetime.now(timezone.utc)
    result = {'after_owner': {
        'before_reminders': [], 'reply_status': 'ready',
        'after_reminders': [{'message': 'call the dentist', 'due_at': (start + timedelta(minutes=5)).isoformat()}],
        'submitted_at': start.isoformat(), 'completed_at': start.isoformat(),
        'offered_before_due': False, 'offered_at_due': True, 'offered_message': 'call the dentist',
    }}
    grade = scenario().grade
    assert grade(result)['effect_passed'] is True
    result['after_owner']['offered_at_due'] = False
    assert grade(result)['effect_passed'] is False
    result['after_owner']['failure_code'] = 'model_unavailable'
    assert grade(result)['effect_passed'] is None
    assert grade({})['effect_passed'] is None
