"""The owner worker keeps an authorized handoff alive after a reminder is blocked."""
from datetime import datetime, timedelta, timezone
import json
import os

from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail_search import OwnerGmailSearchService
from alphonse.agent_v2.core.intelligence.simple_loop.owner_work_log import owner_work_log
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from dojo_mixed_followup_fixture import DisposableReceiver, attach_disposable_receiver
from test_owner_calendar import calendar_worker, worker_status
from test_owner_gmail_search_conversation import BODY, f, inbox_fixture, seed_mail


def test_blocked_reminder_leaves_independent_handoff_available(f, tmp_path):
    seed_mail(f)
    wire = inbox_fixture()
    receiver = DisposableReceiver(tmp_path / 'receiver')
    target, revision = 'workshop-preview', 'fixture-revision'
    due = (datetime.now(timezone.utc) + timedelta(days=3)).replace(microsecond=0).isoformat()
    note = 'The private workshop name is Orchid Compass.'
    followup = (
        f'{note} Ask Codex to check {target} and propose a patch if clipping reproduces. '
        f'Please remind me to run the test print at {due}.'
    )
    decisions = [
        ('native.owner_gmail_search', {'query': 'workshop'}),
        (None, 'The workshop starts at 11:30. [1]'),
        ('native.owner_clock', {}),
        ('native.owner_reminder_create', {
            'message': 'run the test print', 'due_at': due, 'request_quote': followup,
        }),
        ('native.owner_disposable_handoff', {'target': target, 'worker': 'codex', 'operation': 'propose_patch'}),
        (None, 'The reminder was blocked. Codex accepted the handoff, and its results are pending.'),
    ]
    requests = []

    class ScriptedProvider:
        supports_image_input = False

        def generate_json(self, request):
            tool, value = decisions[len(requests)]
            requests.append(request.prompt)
            action = {'kind': 'tool' if tool else 'answer', 'tool_id': tool,
                      'arguments': value if tool else None, 'text': None if tool else value}
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 45})

    def configure(runtime):
        runtime.owner_gmail_search_service = OwnerGmailSearchService(
            f.accounts, lambda _: f.service, transport=wire,
        )
        attach_disposable_receiver(runtime, actor='owner', target=target, revision=revision,
                                   prompt=followup, receiver=receiver)

    def deliver(daemon, queue, prompt):
        message_id = queue.enqueue_trusted_local_owner(prompt=prompt, peer_uid=os.getuid())
        assert worker_status(lambda: daemon.owner_message_status(message_id)) == 'ready'
        draft = daemon.take_owner_desktop_draft(message_id)
        daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        assert daemon.owner_message_status(message_id) == 'client_acknowledged'
        return message_id, draft

    with calendar_worker(f, tmp_path, wire, ScriptedProvider(), configure_runtime=configure) as (daemon, queue, _, _):
        _, digest = deliver(daemon, queue, 'What should I pay attention to in my Inbox?')
        assert digest['private_gmail'] is True
        message_id, draft = deliver(daemon, queue, followup)
        log = owner_work_log(queue, message_id)
        assert OwnerDesktopReminders(queue.db_path).list_open('owner') == []

    assert len(requests) >= 5
    assert 'Gmail Inbox matches were displayed privately in Desktop.' in requests[2]
    assert all(BODY not in request for request in requests[2:])
    receipts = receiver.receipts()
    assert len(receipts) == 1
    assert receipts[0]['status'] == 'accepted'
    brief = receipts[0]['request']
    assert set(brief) == {'operation_id', 'worker', 'target', 'revision', 'operation'}
    assert (brief['worker'], brief['target'], brief['revision'], brief['operation']) == (
        'codex', target, revision, 'propose_patch',
    )
    assert BODY not in json.dumps(brief) and note not in json.dumps(brief)
    assert 'reminder' in draft['message'].lower()
    assert 'codex' in draft['message'].lower()
    assert log['status'] == 'completed'
    outcomes = [(event['label'], event['outcome']) for event in log['events']]
    assert ('Checked the reminder request', 'Request blocked; no reminder created') in outcomes
    assert ('Requested a Codex handoff', 'Worker accepted; result pending') in outcomes
    assert outcomes.index(('Checked the reminder request', 'Request blocked; no reminder created')) < (
        outcomes.index(('Requested a Codex handoff', 'Worker accepted; result pending'))
    )
