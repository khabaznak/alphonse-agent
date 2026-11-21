"""Disposable owner-loop Dojo using real-model trusted host Bash.

WARNING: The fixture is disposable but native Bash has the daemon user's full
host and network access. This is not a sandbox or a household security test.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sqlite3
from pathlib import Path
import tempfile
import time

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.evaluation import dojo_lead_followup
from alphonse.agent_v2.evaluation import dojo_host_recovery
from alphonse.agent_v2.evaluation.owner_gym_smoke import provider_failure_category
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def run(model: str, *, case: str = 'prospect-followup') -> dict:
    calls = {'started': 0, 'returned': 0, 'bash_decisions': 0, 'failures': []}
    if case not in {'prospect-followup', 'host-unavailable-maintenance-report'}:
        raise ValueError('trusted_host_dojo_case_invalid')
    scenario = dojo_host_recovery if case == dojo_host_recovery.CASE_ID else dojo_lead_followup
    with tempfile.TemporaryDirectory(prefix='alphonse-host-dojo-', dir='/private/tmp') as temp:
        root = Path(temp)
        path = root / 'dojo.sqlite'
        users = V2UserStore(path)
        users.set_workspace_root(root / 'workspace')
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        workspace = scenario.seed(users.ordinary_workspace(owner.user_id))
        queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(path)
        router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=model))
        router.provider.config = replace(router.provider.config, timeout_seconds=40)
        generate = router.generate_json

        def observe(request):
            calls['started'] += 1
            try:
                result = generate(request)
            except Exception as exc:
                calls['failures'].append(provider_failure_category(exc))
                raise
            calls['returned'] += 1
            if isinstance(result.json_value, dict) and result.json_value.get('tool_id') == 'native.bash':
                calls['bash_decisions'] += 1
            return result

        router.generate_json = observe
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
            conversation_store=conversation, outbox=SQLiteOutboundStore(path), inference=router,
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)
        daemon = V2Daemon(runtime, daemon_id='trusted-host-dojo', experimental_owner_only=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
        status, draft, acknowledged = 'not_started', None, False
        try:
            daemon.start()
            client = V2DaemonClient(root / 'owner.sock')
            identity = client.runtime_identity()
            message_id = client.request('experimental_owner_ingress', prompt=scenario.goal(workspace))['message_id']
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                status = client.request('experimental_owner_status', message_id=message_id)['status']
                if status not in {'pending', 'processing'}:
                    break
                time.sleep(.1)
            if status == 'ready':
                draft = client.request('experimental_owner_take_draft', message_id=message_id)
                client.request('experimental_owner_ack_draft', message_id=message_id,
                    acknowledgement_token=draft['acknowledgement_token'])
                acknowledged = True
        finally:
            daemon.stop()
        # Inspect only process-result categories; raw commands and output can
        # contain host data, even though the challenge fixture is disposable.
        process_results = []
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
            row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                               (message_id,)).fetchone()
            if row:
                metadata = json.loads(row[0]).get('metadata', {})
                for observation in metadata.get('owner_read_observations', {}).values():
                    result = observation.get('host_bash_observation') if isinstance(observation, dict) else None
                    if isinstance(result, dict):
                        entry = {'exit_code': result.get('exit_code'), 'cancelled': result.get('cancelled'),
                                 'timed_out': result.get('timed_out')}
                        if scenario is dojo_host_recovery:
                            output = str(result.get('stdout') or '') + '\n' + str(result.get('stderr') or '')
                            entry['synthetic_reading_in_output'] = '53.5' in output
                            entry['synthetic_technician_note_in_output'] = 'Technician observed incoming water pressure' in output
                        process_results.append(entry)
        readback = scenario.readback(workspace)
        grade = (scenario.grade(readback, draft.get('message') if draft else None)
                 if scenario is dojo_host_recovery else scenario.grade(readback))
        return {'case': case, 'mode': identity.get('owner_host_bash_mode'),
                'loop': identity.get('loop'), 'status': status, 'client_acknowledged': acknowledged,
                'model_calls': calls, 'effect_passed': grade.get('effect_passed'), 'checks': grade.get('checks'),
                'host_bash_process_results': process_results,
                'host_bash_process_results_order': 'unspecified',
                'target_status': grade.get('target_status'),
                'answer_available': bool(draft and draft.get('message')),
                'scenario_assessment': grade if scenario is dojo_host_recovery else None,
                'limitations': ['Full host and network access, not a sandbox.',
                    'Readback checks fixture effects, not all possible host effects.',
                    'One attempt is not a reliability estimate.', 'Response semantics are unassessed.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--case', choices=('prospect-followup', 'host-unavailable-maintenance-report'),
                        default='prospect-followup')
    parser.add_argument('--accept-trusted-host', action='store_true')
    args = parser.parse_args()
    if not args.accept_trusted_host:
        parser.error('unrestricted host Bash requires --accept-trusted-host')
    with args.journal.open('x', encoding='utf-8') as stream:
        json.dump({'case': args.case, 'status': 'running'}, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        try:
            result = run(args.model, case=args.case)
        except Exception as exc:
            result = {'case': args.case, 'status': 'runner_error', 'error_type': type(exc).__name__}
            stream.seek(0)
            stream.truncate()
            json.dump(result, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            raise
        stream.seek(0)
        stream.truncate()
        json.dump(result, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
