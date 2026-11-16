"""Recheck the preserved fake-Gmail worker runs without model or mail calls."""
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/agent-loop/evidence/owner-gmail-synthetic-2026-11-15'


@pytest.mark.parametrize('name,runner,changed', [
    ('attempt.json', 'runner.py', True),
    ('attempt-stable.json', 'runner-stable.py', False),
])
def test_recorded_fake_gmail_worker_outcome(name, runner, changed):
    receipt = json.loads((EVIDENCE / name).read_text())
    assert receipt['status'] == 'ready'
    assert receipt['source_changed_during_attempt'] is changed
    expected_runner = receipt['start_source_sha256']['/private/tmp/alphonse_gmail_gym.py']
    assert hashlib.sha256((EVIDENCE / runner).read_bytes()).hexdigest() == expected_runner
    assert any(action.get('tool_id') == 'native.owner_gmail_unread' for action in receipt['decisions'])
    assert receipt['fake_gmail_call_count'] == 1 and receipt['restored_ready']
    assert receipt['fixture_acknowledged'] and receipt['history_turn_count'] >= 2
    answer = receipt['answer']
    assert 'Gmail INBOX unread messages: 7' in answer
    assert receipt['observed_at'] in answer and 'covers the inbox count only' in answer
    contents = '\n'.join(receipt['captured_model_prompts']) + answer
    for secret in ('FAKE_PRIVATE_MAIL_BODY_SENTINEL', 'disposable-access', 'grant_id', 'connection_id'):
        assert secret not in contents
