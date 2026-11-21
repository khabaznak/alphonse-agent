from alphonse.agent_v2.evaluation.dojo_ambiguous_reminder import scenario


def test_ambiguous_reminder_does_not_count_scheduled_effect_as_success():
    case = scenario('es')
    clear = {'runtime': {'open_reminders': [], 'scheduled_tasks': []}}
    assert case.grade(clear)['no_scheduled_effect'] is True
    scheduled = {'runtime': {'open_reminders': [], 'scheduled_tasks': [{'id': 'synthetic'}]}}
    assert case.grade(scheduled)['no_scheduled_effect'] is False
    assert case.grade({})['effect_passed'] is None
