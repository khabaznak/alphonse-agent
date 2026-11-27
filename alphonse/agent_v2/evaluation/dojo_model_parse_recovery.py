"""Run disposable owner IPC through a command timeout and a model parse failure."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import ModelProfile, ModelResponseParseError
from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop import owner_host_bash
from alphonse.agent_v2.core.intelligence.simple_loop.router_candidate_chooser import RouterCandidateChooser
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import _chooser_failure_category
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.evaluation.dojo_failure_diagnosis import _wait
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.skill_bindings import ExecutionBindings
from alphonse.agent_v2.users import V2UserStore

CASES = ('recovers', 'malformed_again', 'tries_to_replay', 'unknown_usage')
ANSWER = 'The local check timed out. The fixture state is still unknown.'
COMMAND = 'printf dojo-fixture'
PRIVATE_OUTPUT = 'Fixture reachability check received a reply. The service check started.'
REJECTED_OUTPUT = 'synthetic invalid arguments {'
FIRST_FAILURE_USAGE = {'input_tokens': 211, 'output_tokens': 17}


def run(case: str, *, timeout: float = 10, model: str | None = None) -> dict:
    if case not in CASES:
        raise ValueError('unknown_model_parse_recovery_case')
    if model is not None and case != 'recovers':
        raise ValueError('live_model_requires_recovers_case')
    started = time.monotonic()
    command_action = json.dumps({'kind': 'tool', 'tool_id': 'native.bash',
        'arguments': json.dumps({'command': COMMAND, 'timeout_seconds': 1}),
        'text': None, 'activity_intent': 'check_state'})
    malformed_action = json.dumps({'kind': 'tool', 'tool_id': 'native.bash',
        'arguments': REJECTED_OUTPUT, 'text': None, 'activity_intent': 'check_state'})
    retry_output = {
        'recovers': json.dumps({'kind': 'answer', 'tool_id': None, 'arguments': None,
                                'text': ANSWER, 'activity_intent': None}),
        'malformed_again': 'synthetic invalid outer response',
        'tries_to_replay': command_action,
        'unknown_usage': command_action,
    }[case]
    outputs = iter(((command_action, {'input_tokens': 200, 'output_tokens': 30}),
                    (malformed_action, {} if case == 'unknown_usage' else FIRST_FAILURE_USAGE),
                    (retry_output, {'input_tokens': 220, 'output_tokens': 20})))
    calls: list[dict] = []
    commands: list[str] = []
    if model is None:
        adapter = OpenAICodexProvider()
        router = InferenceRouter(provider=adapter, default_profile=ModelProfile('fixture', 'scripted', 'offline'))
    else:
        router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=model))
        adapter = router.provider
        adapter.config = replace(adapter.config, timeout_seconds=55.0)
    real_transport = adapter._run_codex_with_usage
    real_choose = RouterCandidateChooser._generate

    def transport(prompt, request):
        live_retry = model is not None and len(calls) == 2
        call = {'source': 'live' if live_retry else 'scripted', 'saw_observation': PRIVATE_OUTPUT in prompt,
                'saw_rejected_output': REJECTED_OUTPUT in prompt}
        calls.append(call)
        if live_retry:
            call['schema_action_kinds'] = request.output_schema['properties']['kind']['enum']
            response, usage = real_transport(prompt, request)
            # This live call sees only the disposable fixture. Keep its reply
            # for diagnosis even when the owner worker declines to deliver it.
            call['response'] = response
        elif model is not None and len(calls) > 3:
            raise RuntimeError('unexpected_model_attempt_after_live_retry')
        else:
            response, usage = next(outputs)
        call['reported_usage'] = usage
        return response, usage

    def choose(chooser, request, output_limit):
        before = len(calls)
        try:
            decision = real_choose(chooser, request, output_limit)
        except Exception as exc:
            if len(calls) > before:
                if type(exc) is ModelResponseParseError:
                    calls[-1]['parse_stage'] = exc.response_stage
                else:
                    calls[-1]['chooser_failure'] = _chooser_failure_category(exc)
            raise
        if len(calls) > before:
            calls[-1]['normalized_kind'] = decision['kind']
        return decision

    def observe_timeout(arguments, *, context, _binding_observer):
        commands.append(arguments['command'])
        if arguments['command'] != COMMAND or context.cancellation_checker():
            raise ValueError('unexpected_fixture_command')
        _binding_observer(ExecutionBindings({}, ()))
        return {'exit_code': -1, 'timed_out': True, 'cancelled': False,
                'stdout': PRIVATE_OUTPUT, 'stderr': 'Synthetic timeout'}

    with tempfile.TemporaryDirectory(prefix='dojo-parse-', dir='/tmp') as directory:
        root = Path(directory)
        db = root / 'owner.sqlite'
        users = V2UserStore(db)
        users.set_users_root(root / 'users')
        users.set_workspace_root(root / 'workspace')
        users.onboard(display_name='Synthetic owner')
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
            conversation_store=SQLiteConversationStore(db), outbox=SQLiteOutboundStore(db), inference=router,
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        runtime.processor.owner_read_candidate.enable_owner_host_bash(allow_unprotected_for_testing=True)
        daemon = V2Daemon(runtime, daemon_id='dojo-model-parse', experimental_owner_only=True)
        daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
        with (patch.object(adapter, '_run_codex_with_usage', transport),
              patch.object(RouterCandidateChooser, '_generate', choose),
              patch.object(owner_host_bash, 'execute_bash', observe_timeout),
              patch.object(daemon, '_acquire_single_instance_lock', lambda: None),
              patch.object(daemon, '_release_single_instance_lock', lambda: None)):
            daemon.start()
            try:
                client = V2DaemonClient(root / 'owner.sock')
                message = client.request('experimental_owner_ingress',
                    prompt='Check the state of the disposable local fixture and tell me what you found.')['message_id']
                status = _wait(client, message, timeout)
                work_log = client.request('experimental_owner_work_log', message_id=message)
                with queue._connect() as conn:
                    row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message,)).fetchone()
                checkpoint = json.loads(row[0])['metadata']['single_loop_checkpoint']
                reservations = checkpoint['budget']['reservations'].values()
                parse_failures = {row['parse_stage']: {
                    'reported_usage': {key: row['actual'][key] for key in FIRST_FAILURE_USAGE
                                       if key not in row['unknown_tokens']},
                    'unknown_usage': row['unknown_tokens'],
                } for row in reservations if 'parse_stage' in row}
                draft = client.request('experimental_owner_take_draft', message_id=message) if status['status'] == 'ready' else None
                acknowledgement = client.request('experimental_owner_ack_draft', message_id=message,
                    acknowledgement_token=draft['acknowledgement_token']).get('status') if draft else None
                return {'case': case, 'requested_model': model, 'status': status, 'work_log': work_log, 'model_calls': calls,
                        'blocker': checkpoint['state']['blocker'],
                        'command_dispatches': len(commands), 'parse_failures': parse_failures,
                        'format_retries': checkpoint['state']['invalid_decision_retries'],
                        'rejected_output_retained': REJECTED_OUTPUT in row[0],
                        'answer': draft['message'] if draft else None, 'acknowledgement': acknowledgement,
                        'elapsed_seconds': round(time.monotonic() - started, 3),
                        'limitations': [
                            'The initial command selection, command timeout, and first parse failure are scripted.',
                            'Only the final retry uses a real model.' if model else 'The retry is scripted.',
                            'The fixture observes owner IPC, without a real shell, device, or Desktop UI.',
                        ]}
            finally:
                daemon.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--model', help='Use this Codex model for the retry in the recovers case.')
    args = parser.parse_args()
    if args.model and args.case != 'recovers':
        parser.error('--model requires --case recovers')
    with args.journal.open('x', encoding='utf-8') as journal:
        try:
            report = run(args.case, model=args.model, timeout=75 if args.model else 10)
        except Exception as exc:
            json.dump({'case': args.case, 'requested_model': args.model, 'status': 'infrastructure_error',
                       'error_type': type(exc).__name__}, journal)
            journal.write('\n')
            raise
        json.dump(report, journal, sort_keys=True)
        journal.write('\n')


if __name__ == '__main__':
    main()
