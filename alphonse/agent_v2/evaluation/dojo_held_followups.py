"""Observe held follow-ups during a real owner-loop request with disposable data.

This scripted plumbing observation does not evaluate model understanding or
queue execution: execution is observed using scripted responses. It never reports
the issue's semantic multi-turn capability as passed.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


class PausedProvider:
    supports_image_input = False

    def __init__(self, fail=False, delegate=None):
        self.fail = fail
        self.delegate = delegate
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.prompts = []
        self.provider_failure = None

    def generate_json(self, request):
        self.calls += 1
        self.prompts.append(request.prompt)
        self.entered.set()
        if not self.release.wait(15):
            raise TimeoutError('fixture was not released')
        if self.fail:
            raise RuntimeError('synthetic provider failure')
        if self.delegate:
            try:
                return self.delegate(request)
            except Exception as exc:
                from alphonse.agent_v2.evaluation.owner_gym_smoke import provider_failure_category
                self.provider_failure = {'category': provider_failure_category(exc), 'type': type(exc).__name__}
                raise
        decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'The list is ready.'}
        return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                               usage={'input_tokens': 100, 'output_tokens': 20})


def run(*, timeout=10, scenario="normal", model=None):
    """Return observations, with unimplemented execution explicitly unassessed."""
    with tempfile.TemporaryDirectory(prefix='dojo-held-followups-') as directory:
        root = Path(directory)
        path = root / 'owner.sqlite'
        users = V2UserStore(path)
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(path)
        if model:
            from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
            router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=model))
            provider = PausedProvider(delegate=router.generate_json)
            router.generate_json = provider.generate_json
        else:
            provider = PausedProvider(fail=scenario == "failure")
            router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'dojo-owner'))
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
            conversation_store=conversation, outbox=SQLiteOutboundStore(path), inference=router,
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        if model is None:
            # Scripted plumbing only; this is not a Jev evaluation.
            from types import SimpleNamespace
            runtime.core.system_one = SimpleNamespace(review_freeform_effect_claim=lambda **_: True)
        daemon = V2Daemon(runtime, daemon_id='dojo-held-followups', experimental_owner_only=True)
        daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.start()
        try:
            client = V2DaemonClient(root / 'owner.sock')
            initial_prompt = ('Suggest one fruit for the shopping list: red apples.' if scenario == 'steer'
                              else 'In this chat only, imagine a shopping list containing two red apples. Do not save or change any files. Tell me the imagined list.' if scenario == 'hypothetical'
                              else 'Plan a simple shopping list.')
            active = client.request('experimental_owner_ingress', prompt=initial_prompt)['message_id']
            if not provider.entered.wait(timeout):
                return {'observation': 'model_not_reached', 'passed': None, 'execution_assessed': False}
            first, removed_id, second = '1' * 32, '2' * 32, '3' * 32
            if scenario == "cancel":
                steer = client.request('experimental_owner_steer_add', active_message_id=active, request_id=first, prompt='Use green apples.')
                queued = client.request('experimental_owner_followup_add', active_message_id=active, request_id=second, prompt='Add pears.')
                cancellation = client.request('trigger_killswitch', actor_user_id=owner.user_id)
                provider.release.set()
                deadline = time.monotonic() + timeout
                status = 'pending'
                while time.monotonic() < deadline:
                    status = client.request('experimental_owner_status', message_id=active).get('status')
                    if status not in ('pending', 'processing'):
                        break
                    time.sleep(.05)
                steering = client.request('experimental_owner_steer_list')['items']
                held = client.request('experimental_owner_followup_list')['items']
                return {'observation': 'cancel', 'cancel_status': cancellation.get('status'),
                        'initial_status': status, 'steer_submitted': steer['status'],
                        'steer_status': steering[0]['status'] if steering else None,
                        'queued': queued['status'], 'held': [(item['id'], item['status']) for item in held],
                        'model_calls': provider.calls, 'passed': None,
                        'limitations': ['Scripted model; semantic behavior is not evaluated.',
                                        'No external side effect was in flight.']}
            if scenario == "steer":
                client.request('experimental_owner_composer_save', id=first, active_message_id=active, prompt='Use green apples.')
                submitted = client.request('experimental_owner_steer_add', active_message_id=active, request_id=first, prompt='Use green apples.')
                provider.release.set()
                deadline = time.monotonic() + timeout
                status = 'pending'
                while time.monotonic() < deadline:
                    status = client.request('experimental_owner_status', message_id=active).get('status')
                    if status not in ('pending', 'processing'):
                        break
                    time.sleep(.05)
                listed = client.request('experimental_owner_steer_list')['items']
                answer = None
                if status == 'ready':
                    draft = client.request('experimental_owner_take_draft', message_id=active)
                    answer = draft['message']
                    client.request('experimental_owner_ack_draft', message_id=active,
                                   acknowledgement_token=draft['acknowledgement_token'])
                late = client.request('experimental_owner_steer_add', active_message_id=active, request_id=second, prompt='Use pears.')
                return {'observation': 'steer', 'submission': submitted['status'], 'initial_status': status,
                        'steering_status': listed[0]['status'] if listed else None,
                        'context_visible': ["Use green apples." in prompt for prompt in provider.prompts],
                        'late_status': late['status'], 'model_calls': provider.calls, 'answer': answer, 'model': model or 'scripted', 'provider_failure': provider.provider_failure, 'synthetic_effect_review': model is None, 'passed': None,
                        'limitations': [
                            'Semantic task outcome requires independent manual review.' if model else 'Scripted model; semantic task outcome is not evaluated.',
                            'Restart and cancellation while steering are not assessed.' ]}
            if scenario == "draft_race":
                from concurrent.futures import ThreadPoolExecutor
                client.request('experimental_owner_composer_save', id=first, active_message_id=active, prompt='Add apples.')
                barrier = threading.Barrier(2)
                def add():
                    barrier.wait()
                    try:
                        return client.request('experimental_owner_followup_add', active_message_id=active,
                                              prompt='Add apples.', request_id=first)['status']
                    except RuntimeError:
                        return 'rejected'
                def reconcile():
                    barrier.wait()
                    return client.request('experimental_owner_composer_reconcile', id=first)['resolution']
                with ThreadPoolExecutor(max_workers=2) as executor:
                    a, b = executor.submit(add), executor.submit(reconcile)
                    add_result, resolution = a.result(), b.result()
                held = client.request('experimental_owner_followup_list')['items']
                return {'observation': 'draft_race', 'add_result': add_result, 'resolution': resolution,
                        'held_ids': [item['id'] for item in held], 'passed': None,
                        'limitations': ['One concurrent socket race; all interleavings are not exhausted.']}
            if scenario == "draft_fence":
                client.request('experimental_owner_composer_save', id=first, active_message_id=active, prompt='Add apples.')
                resolution = client.request('experimental_owner_composer_reconcile', id=first)
                try:
                    client.request('experimental_owner_followup_add', active_message_id=active, prompt='Add apples.', request_id=first)
                except RuntimeError:
                    rejected = True
                else:
                    rejected = False
                held = client.request('experimental_owner_followup_list')['items']
                return {'observation': 'draft_fence', 'resolution': resolution['resolution'],
                        'late_add_rejected': rejected, 'held_ids': [item['id'] for item in held],
                        'passed': None, 'limitations': ['Ordered IPC calls; concurrent socket race not simulated.']}
            if scenario == "draft_restart":
                client.request('experimental_owner_composer_save', id=first, active_message_id=active, prompt='Add apples.')
                # Ignore the add response to model an interrupted client after commit.
                client.request('experimental_owner_followup_add', active_message_id=active, prompt='Add apples.', request_id=first)
                provider.release.set()
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if client.request('experimental_owner_status', message_id=active).get('status') == 'ready':
                        break
                    time.sleep(.05)
                daemon.stop()
                daemon = V2Daemon(runtime, daemon_id='dojo-draft-restarted', experimental_owner_only=True)
                daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
                daemon._acquire_single_instance_lock = lambda: None
                daemon._release_single_instance_lock = lambda: None
                daemon.start()
                client = V2DaemonClient(root / 'owner.sock')
                saved = client.request('experimental_owner_composer_list')['items']
                resolution = client.request('experimental_owner_composer_reconcile', id=first)
                held = client.request('experimental_owner_followup_list')['items']
                return {'observation': 'draft_restart', 'saved_ids': [item['id'] for item in saved],
                        'resolution': resolution['resolution'], 'held_ids': [item['id'] for item in held],
                        'model_calls': provider.calls, 'passed': None,
                        'limitations': ['Synthetic discarded response; no interrupted socket write simulated.',
                                        'Scripted model; semantic behavior is not evaluated.']}
            first_prompt = ('In that imagined list, add two apples. Answer in chat only; do not save or change files.'
                            if scenario == 'hypothetical' else 'Add apples.')
            second_prompt = ('In that imagined list, make all apples green. Answer in chat only; do not save or change files.'
                             if scenario == 'hypothetical' else 'Make those apples green.')
            added = [client.request('experimental_owner_followup_add', active_message_id=active,
                                    prompt=prompt, request_id=item)
                     for item, prompt in ((first, first_prompt), (removed_id, 'Add onions.'),
                                          (second, second_prompt))]
            listed = client.request('experimental_owner_followup_list')['items']
            removed = client.request('experimental_owner_followup_remove', id=removed_id)
            before_ack = client.request('experimental_owner_followup_list')['items']
            provider.release.set()

            def wait_ready(message_id):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    state = client.request('experimental_owner_status', message_id=message_id)
                    if state.get('status') == 'ready':
                        return True
                    time.sleep(.05)
                return False

            def acknowledge(message_id):
                draft = client.request('experimental_owner_take_draft', message_id=message_id)
                client.request('experimental_owner_ack_draft', message_id=message_id,
                               acknowledgement_token=draft['acknowledgement_token'])
                return draft['message']

            if scenario == "failure":
                deadline = time.monotonic() + timeout
                status = 'pending'
                while time.monotonic() < deadline:
                    status = client.request('experimental_owner_status', message_id=active).get('status')
                    if status == 'failed':
                        break
                    time.sleep(.05)
                time.sleep(.5)
                held = client.request('experimental_owner_followup_list')['items']
                return {'observation': 'failed_predecessor', 'initial_status': status,
                        'held': [(item['id'], item['status']) for item in held],
                        'model_calls': provider.calls, 'passed': None,
                        'limitations': ['Scripted provider failure; no semantic behavior assessed.',
                                        'Cancellation not assessed.']}
            initial_ready = wait_ready(active)
            if scenario == "uncertain":
                if initial_ready:
                    client.request('experimental_owner_take_draft', message_id=active)
                daemon.stop()
                daemon = V2Daemon(runtime, daemon_id='dojo-held-restarted', experimental_owner_only=True)
                daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
                daemon._acquire_single_instance_lock = lambda: None
                daemon._release_single_instance_lock = lambda: None
                daemon.start()
                client = V2DaemonClient(root / 'owner.sock')
                time.sleep(.5)
                held = client.request('experimental_owner_followup_list')['items']
                return {'observation': 'uncertain_restart', 'initial_ready': initial_ready,
                        'held': [(item['id'], item['status']) for item in held],
                        'model_calls': provider.calls, 'passed': None,
                        'limitations': ['Scripted model; semantic behavior not assessed.',
                                        'No cancellation or failure scenario assessed.']}
            initial_answer = acknowledge(active) if initial_ready else None
            first_ready = wait_ready(first)
            before_first_ack = client.request('experimental_owner_followup_list')['items']
            first_answer = acknowledge(first) if first_ready else None
            second_ready = wait_ready(second)
            second_answer = acknowledge(second) if second_ready else None
            after = client.request('experimental_owner_followup_list')['items']
            with queue._connect() as conn:
                ingress_count = conn.execute('SELECT count(*) FROM v2_inbound_messages').fetchone()[0]
            return {'observation': 'followup_execution', 'initial_ready': initial_ready,
                    'first_ready': first_ready, 'second_ready': second_ready,
                    'accepted_statuses': [item['status'] for item in added],
                    'listed_ids': [item['id'] for item in listed],
                    'remove_status': removed['status'],
                    'before_ack': [(item['id'], item['status']) for item in before_ack],
                    'before_first_ack': [(item['id'], item['status']) for item in before_first_ack],
                    'after': [(item['id'], item['status']) for item in after],
                    'inbound_count': ingress_count, 'model_calls': provider.calls,
                    'answers': [initial_answer, first_answer, second_answer], 'model': model or 'scripted',
                    'provider_failure': provider.provider_failure,
                    'context_visible': [all(text in prompt for text in (first_prompt, second_prompt))
                                        for prompt in provider.prompts],
                    'passed': None, 'semantic_execution_assessed': False, 'synthetic_effect_review': model is None,
                    'limitations': [('Semantic behavior requires independent review.' if model else 'Scripted model; semantic behavior is not evaluated.'),
                                    'Steering, restart, and uncertain delivery are not assessed in this case.']}

        finally:
            provider.release.set()
            daemon.stop()
