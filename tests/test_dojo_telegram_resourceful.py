from alphonse.agent_v2.evaluation.dojo_telegram_resourceful import FakeTelegramClient, grade, seed


def test_fixture_grades_real_report_and_detects_wrong_inclusion(tmp_path):
    folder = tmp_path / 'leads'
    prompt = seed(folder)
    assert str(folder) in prompt
    assert 'Mira' not in prompt and 'Cato' not in prompt
    missing = grade(folder, None, [])
    assert missing['report_exists'] is False
    (folder / 'follow-up.md').write_text('Mira\nCato\n2 follow-ups\n', encoding='utf-8')
    result = grade(folder, 'Done', [{'exit_code': 0}])
    assert result['follow_up_names_present'] and result['count_two_mentioned']
    assert result['source_unchanged'] and not result['closed_name_mentioned']
    assert result['report_content'] == 'Mira\nCato\n2 follow-ups\n'
    (folder / 'follow-up.md').write_text('Mira, Cato, Emil: 3', encoding='utf-8')
    assert grade(folder, None, [])['closed_name_mentioned']


def test_fake_telegram_does_not_contact_a_real_service():
    fake = FakeTelegramClient('hello')
    assert fake.get_updates(offset=None, timeout=0)[0]['message']['text'] == 'hello'
    assert fake.get_updates(offset=2, timeout=0) == []
    assert fake.send_message(chat_id='123', text='response') == 'synthetic-provider-ack'
    assert fake.sent == [{'chat_id': '123', 'text': 'response'}]
