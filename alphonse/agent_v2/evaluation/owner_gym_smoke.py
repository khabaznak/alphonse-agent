"""One disposable end-to-end smoke obstacle for the owner worker.

By default inference is scripted; explicit model runs use Codex. Both use a
temporary home/database and do not establish the installed application's state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore

PROMPT = 'What is 17 + 25? Reply with only the number.'
CLOCK_PROMPT = ('Read the current UTC time using the clock. Reply with exactly the clock observation\'s '
                'observed_at_utc value, with no other text.')
DOJO_CLOCK_PROMPT = 'What is the current UTC time? Reply with an ISO 8601 timestamp including seconds.'


def provider_failure_category(exc):
    """Keep only known categories; provider errors can contain private text."""
    from alphonse.agent_v2.core.inference.router import _simple_loop_failure_category
    category = _simple_loop_failure_category(exc)
    if category != 'provider_error':
        return category
    if type(exc) is ValueError and len(exc.args) == 1 and type(exc.args[0]) is str:
        code = exc.args[0].split(':', 1)[0]
        if code in {'openai_codex_model_access_rejected', 'openai_codex_exec_failed'}:
            return code
    return 'provider_error'


class ScriptedProvider:
    supports_image_input = False

    def __init__(self, answer='42'):
        self.answer = answer
        self.calls = 0

    def generate_json(self, request):
        self.calls += 1
        value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': self.answer}
        return InferenceResult(json_value=value, raw_response=json.dumps(value),
                               usage={'input_tokens': 100, 'output_tokens': 20})


def grade(status, draft, acknowledged, history):
    """Score the observed draft and recorded history, not the provider's claim."""
    if status != 'ready' or draft is None:
        return {'passed': False if status == 'failed' else None,
                'outcome': 'failed' if status == 'failed' else 'timeout' if status in {'pending', 'processing'} else 'unavailable',
                'reason': 'no_completed_draft'}
    checks = {'answer_kind': draft.get('kind') == 'answer',
              'exact_answer': draft.get('message', '').strip() == '42',
              'client_acknowledged': acknowledged,
              'history_recorded': bool(history) and history[-1].content == draft.get('message')}
    return {'passed': all(checks.values()), 'outcome': 'passed' if all(checks.values()) else 'failed', 'checks': checks}


def grade_clock(status, draft, acknowledged, history, observations, started_at, ended_at):
    if status != 'ready' or draft is None:
        return grade(status, draft, acknowledged, history)
    clocks = [item.get('clock_observation') for item in observations.values()
              if isinstance(item, dict) and item.get('tool_id') == 'native.owner_clock']
    clock = clocks[-1] if clocks else None
    try:
        stamp = datetime.fromisoformat(clock['observed_at_utc'])
        time_valid = (stamp.tzinfo == timezone.utc and started_at <= stamp <= ended_at
                      and clock['timezone'] == 'UTC' and clock['local_time'] == stamp.isoformat())
    except (TypeError, ValueError, KeyError):
        time_valid = False
    message = draft.get('message', '')
    answer, separator, footer = message.partition('\n\nHost clock observed as of: ')
    expected_footer = (f"{clock['local_time']} (UTC; UTC {clock['observed_at_utc']})" if time_valid else None)
    checks = {'answer_kind': draft.get('kind') == 'answer', 'clock_read': bool(clocks),
              'observation_in_run': time_valid,
              'answer_matches_observation': bool(time_valid and answer.strip() == clock['observed_at_utc']),
              'display_matches_observation': bool(separator and footer == expected_footer),
              'client_acknowledged': acknowledged,
              'history_recorded': bool(history) and history[-1].content == message}
    return {'passed': all(checks.values()), 'outcome': 'passed' if all(checks.values()) else 'failed',
            'checks': checks, 'clock_observation': clock}


def grade_dojo_clock(status, draft, acknowledged, history, observations, started_at, ended_at):
    """Assess the answer against the harness clock without prescribing a tool."""
    if status != 'ready' or draft is None:
        return grade(status, draft, acknowledged, history)
    message = draft.get('message', '')
    answer = message.split('\n\nHost clock observed as of: ', 1)[0].strip()
    try:
        instant = datetime.fromisoformat(answer.replace('Z', '+00:00'))
        utc_answer = instant.tzinfo is not None and instant.utcoffset().total_seconds() == 0
        # A timestamp with second precision may be truncated, rather than rounded.
        in_window = utc_answer and started_at - timedelta(seconds=1) <= instant <= ended_at
        has_seconds = bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)', answer))
    except (TypeError, ValueError, AttributeError, OverflowError):
        in_window, has_seconds = False, False
    checks = {'answer_kind': draft.get('kind') == 'answer', 'utc_timestamp_with_seconds': has_seconds,
              'time_in_request_window': bool(in_window), 'client_acknowledged': acknowledged,
              'history_recorded': bool(history) and history[-1].content == message}
    observed_tools = [item.get('tool_id') for item in observations.values()
                      if isinstance(item, dict) and isinstance(item.get('tool_id'), str)]
    return {'passed': all(checks.values()), 'outcome': 'passed' if all(checks.values()) else 'failed',
            'checks': checks, 'observed_tools': observed_tools,
            'pillars': {'resourcefulness_improvisation': 'observed_host_tool_only' if observed_tools else 'unassessed',
                        'resilience': 'unassessed', 'accountability': 'partial_delivery_and_history',
                        'human_understanding': 'one_request_only', 'trustworthiness': 'unassessed'},
            'limitations': ['Host-tool records do not capture any tools used inside the inference provider.',
                            'Correct time alone does not establish grounding or host clock accuracy.',
                            'No recovery, external effect, alternative route or authorization challenge was tested.']}


def run(provider=None, *, router=None, timeout_seconds=10, case='arithmetic'):
    """Run in disposable storage through local socket, daemon and simple loop."""
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='alphonse-gym-') as directory:
        root = Path(directory)
        env = {k: v for k, v in os.environ.items() if not k.startswith('ALPHONSE_')}
        env['ALPHONSE_V2_DB_PATH'] = str(root / 'gym.sqlite')
        env['ALPHONSE_V2_SOCKET_PATH'] = str(root / 'gym.sock')
        with patch.dict(os.environ, env, clear=True), patch.object(Path, 'home', classmethod(lambda cls: root)):
            path = root / 'gym.sqlite'
            users = V2UserStore(path)
            owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
            users.set_timezone('UTC', owner.user_id)
            queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
            conversation = SQLiteConversationStore(path)
            tools = InMemoryToolRegistry()
            tools.register(build_search_memory_tool_definition())
            tools.register(build_read_memory_source_tool_definition())
            if router is None:
                router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'gym-smoke'))
            calls = {'started': 0, 'returned': 0}
            provider_failure = []
            original_generate = router.generate_json
            def counted_generate(request):
                calls['started'] += 1
                try:
                    result = original_generate(request)
                except Exception as exc:
                    provider_failure.append(provider_failure_category(exc))
                    raise
                calls['returned'] += 1
                return result
            router.generate_json = counted_generate
            runtime = build_runtime_host(user_store=users, messages=queue,
                question_store=SQLiteQuestionStore(path), conversation_store=conversation,
                outbox=SQLiteOutboundStore(path), tools=tools, inference=router,
                coding_skill_service=object(), experimental_owner_read=True,
                owner_selected_draft_sink=lambda *_: None)
            daemon = V2Daemon(runtime, daemon_id='gym-smoke', experimental_owner_only=True)
            daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'gym.sock')
            status, draft, acknowledged = 'not_started', None, False
            try:
                daemon.start()
                client = V2DaemonClient(root / 'gym.sock')
                started_at = datetime.now(timezone.utc)
                prompts = {'clock': CLOCK_PROMPT, 'dojo-clock': DOJO_CLOCK_PROMPT, 'arithmetic': PROMPT}
                message_id = client.request('experimental_owner_ingress', prompt=prompts[case])['message_id']
                deadline = time.monotonic() + timeout_seconds
                status = 'pending'
                status_payload = {}
                while time.monotonic() < deadline:
                    status_payload = client.request('experimental_owner_status', message_id=message_id)
                    status = status_payload['status']
                    if status not in {'pending', 'processing'}:
                        break
                    time.sleep(.02)
                observations = {}
                if status == 'ready':
                    if case in {'clock', 'dojo-clock'}:
                        checkpoint, _ = daemon._owner_checkpoints[message_id]
                        _, observations = checkpoint.load(expected_version=checkpoint.version)
                    draft = client.request('experimental_owner_take_draft', message_id=message_id)
                    client.request('experimental_owner_ack_draft', message_id=message_id,
                                   acknowledgement_token=draft['acknowledgement_token'])
                    acknowledged = client.request('experimental_owner_status', message_id=message_id)['status'] == 'client_acknowledged'
                history = conversation.list(owner_user_id=owner.user_id)
                ended_at = datetime.now(timezone.utc)
                if case == 'clock':
                    score = grade_clock(status, draft, acknowledged, history, observations, started_at, ended_at)
                elif case == 'dojo-clock':
                    score = grade_dojo_clock(status, draft, acknowledged, history, observations, started_at, ended_at)
                else:
                    score = grade(status, draft, acknowledged, history)
                if status == 'failed' and status_payload.get('failure_code') == 'model_unavailable':
                    score = {'passed': None, 'outcome': 'infrastructure_error', 'reason': 'model_unavailable'}
                elif status == 'failed' and status_payload.get('failure_code') == 'time_limit':
                    score = {'passed': False, 'outcome': 'timeout', 'reason': 'task_deadline'}
                if case == 'dojo-clock' and 'pillars' not in score:
                    score['pillars'] = dict.fromkeys(('resourcefulness_improvisation', 'resilience',
                        'accountability', 'human_understanding', 'trustworthiness'), 'unassessed')
                return {'case': {'clock': 'clock-utc', 'dojo-clock': 'dojo-current-utc', 'arithmetic': 'arithmetic-17-plus-25'}[case], 'status': status, **score,
                        'answer': draft.get('message') if draft else None,
                        'request_started_at_utc': started_at.isoformat(), 'request_finished_at_utc': ended_at.isoformat(),
                        'failure_code': status_payload.get('failure_code'),
                        'provider_failure_categories': provider_failure,
                        'provider_calls': calls['started'], 'provider_calls_returned': calls['returned'],
                        'elapsed_seconds': round(time.monotonic() - start, 3)}
            finally:
                daemon.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path, help='new JSONL path; never overwrites')
    parser.add_argument('--model', help='configured Codex model; enables one real-model attempt')
    parser.add_argument('--case', choices=('arithmetic', 'clock', 'dojo-clock'), default='arithmetic')
    args = parser.parse_args()
    router = None
    if args.model:
        from dataclasses import replace
        from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
        router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
        router.provider.config = replace(router.provider.config, timeout_seconds=40.0)
    root = Path(__file__).resolve().parents[3]
    tracked = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=False)
    files = sorted((root / 'alphonse' / 'agent_v2').rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as journal:
        def record(row):
            journal.write(json.dumps(row, sort_keys=True) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        record({'event': 'started', 'case': {'clock': 'clock-utc', 'dojo-clock': 'dojo-current-utc', 'arithmetic': 'arithmetic-17-plus-25'}[args.case],
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'inference': 'openai_codex' if router else 'scripted', 'model': args.model,
                'git_head': tracked.stdout.strip() or None, 'python_source_tree_sha256': digest.hexdigest(),
                'python': sys.version.split()[0], 'platform': platform.platform(),
                'scope': 'disposable owner daemon, socket and data; not installed Desktop',
                'limitations': ['Python source hash excludes non-Python configuration and dependencies.',
                                'Fixture registry, home and storage differ from release.',
                                'Real-model runs set a 40-second provider timeout.']})
        fd = os.open(args.journal.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            result = run(ScriptedProvider() if router is None else None, router=router,
                         timeout_seconds=90 if router else 10, case=args.case)
        except Exception as exc:
            record({'event': 'finished', 'passed': None, 'outcome': 'infrastructure_error', 'status': 'runner_error',
                    'error_type': type(exc).__name__})
            raise
        record({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
