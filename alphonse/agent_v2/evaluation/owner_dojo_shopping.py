"""Owner-daemon shopping Dojo with an isolated Bash service.

By default this is a scripted plumbing test. An explicit model opts into live
Codex inference, separately from Bash's disposable container. The sandbox reduces
exposure but is not a security boundary for the separate inference provider.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import tempfile
import threading
import time
from unittest.mock import patch
from uuid import uuid4

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.evaluation.dojo_bash import DojoBash
from alphonse.agent_v2.evaluation.dojo_household_shopping import GOAL, _readback, _seed, grade
from alphonse.agent_v2.evaluation.owner_gym_smoke import provider_failure_category
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


class ScriptedProvider:
    """The expected fixture command is fixed; no generated host command runs."""
    supports_image_input = False

    def __init__(self):
        self.calls = 0
        self.saw_bash_catalog = False

    def generate_json(self, request):
        self.calls += 1
        self.saw_bash_catalog |= 'native.bash' in request.prompt
        if self.calls == 1:
            program = ('import json,pathlib; p=pathlib.Path("/workspace/household"); '
                       'a=json.loads((p/"pantry.json").read_text()); '
                       'b=json.loads((p/"meal-plan.json").read_text()); '
                       'out={k:v-a.get(k,0) for k,v in b.items() if v>a.get(k,0)}; '
                       'pathlib.Path("/workspace/shopping-list.json").write_text(json.dumps(out))')
            decision = {'kind': 'tool', 'tool_id': 'native.bash',
                        'arguments': {'command': 'python3 -c ' + shlex.quote(program)}, 'text': None}
        else:
            decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'The list is ready.'}
        return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                               usage={'input_tokens': 100, 'output_tokens': 20})


def _capability_outcome(status, failure_code, decisions, observations, effect_passed, provider_failures=(),
                        authority_events=()):
    # A requested Bash action may be blocked before dispatch; the fixture's
    # missing file stays a factual failure, but does not grade an unrun tool.
    blocked = (status == 'failed' and failure_code in {'usage_unconfirmed', 'usage_limit_exceeded'}
               and not observations and any(item.get('kind') == 'tool' and item.get('tool_id') == 'native.bash'
                                            for item in decisions))
    if blocked:
        return {'capability_outcome': 'unassessed_blocked_before_bash', 'task_passed': None}
    if provider_failures:
        return {'capability_outcome': 'unassessed_provider_failure', 'task_passed': None}
    if (status == 'failed' and not observations and any(
            isinstance(event, dict) and event.get('event') == 'authority_denied' for event in authority_events)):
        return {'capability_outcome': 'unassessed_authority_denied_before_bash', 'task_passed': None}
    return {'capability_outcome': 'effect_failed' if effect_passed is False else 'unassessed',
            'task_passed': False if effect_passed is False else None}


def run(*, image_id: str, docker_socket: str, emit, model: str | None = None,
        timeout_seconds: float = 15, scenario=None, full_registry: bool = False,
        synthetic_authority: bool = False, deny_authority_dispatch: bool = False,
        synthetic_clarification_review: bool = False, socket_authority: bool = False,
        synthetic_web_base_url: str | None = None):
    """Exercise local owner IPC and independently inspect the synthetic effect."""
    if type(socket_authority) is not bool or (synthetic_authority and socket_authority):
        raise ValueError('dojo_authority_configuration_invalid')
    if deny_authority_dispatch and not (synthetic_authority or socket_authority):
        raise ValueError('denial_requires_synthetic_authority')
    if scenario is not None and not model:
        raise ValueError('scenario_requires_explicit_model')
    if synthetic_web_base_url is not None:
        from urllib.parse import urlsplit
        parsed = urlsplit(synthetic_web_base_url)
        if (not full_registry or parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
                or parsed.username or parsed.password or parsed.path not in ('', '/')
                or parsed.query or parsed.fragment or parsed.port is None):
            raise ValueError('dojo_synthetic_web_settings_invalid')
    goal = scenario.GOAL if scenario is not None else GOAL
    seed = scenario.seed if scenario is not None else _seed
    readback_effect = scenario.readback if scenario is not None else _readback
    grade_effect = scenario.grade if scenario is not None else grade
    case_id = scenario.CASE_ID if scenario is not None else 'owner-shopping'
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    provider = None if model else ScriptedProvider()
    observations = []
    decisions = []
    provider_failures = []
    authority_events = []
    model_calls = {'started': 0, 'returned': 0, 'bash_advertised': None, 'advertised_tools': None}
    with tempfile.TemporaryDirectory(prefix='owner-dojo-') as directory:
        root = Path(directory)
        env = {key: value for key, value in os.environ.items() if not key.startswith('ALPHONSE_')}
        env['ALPHONSE_V2_DB_PATH'] = str(root / 'fixture.sqlite')
        env['ALPHONSE_V2_SOCKET_PATH'] = str(root / 'owner.sock')
        with patch.dict(os.environ, env, clear=True), patch.object(Path, 'home', classmethod(lambda cls: root)):
            path = root / 'fixture.sqlite'
            users = V2UserStore(path)
            owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
            users.set_timezone('UTC', owner.user_id)
            queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
            conversation = SQLiteConversationStore(path)
            tools = None
            if not full_registry:
                tools = InMemoryToolRegistry()
                tools.register(build_search_memory_tool_definition())
                tools.register(build_read_memory_source_tool_definition())
                tools.register(build_bash_tool_definition())
            if model:
                router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=model))
                router.provider.config = replace(router.provider.config, timeout_seconds=40)
            else:
                router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'dojo-owner'))
            original_generate = router.generate_json
            def observe_model(request):
                model_calls['started'] += 1
                marker = 'Alphonse has the following host tools available: '
                try:
                    catalog, _ = json.JSONDecoder().raw_decode(request.prompt.split(marker, 1)[1])
                    if isinstance(catalog, list):
                        model_calls['advertised_tools'] = [item['tool_id'] for item in catalog
                            if isinstance(item, dict) and isinstance(item.get('tool_id'), str)]
                        model_calls['bash_advertised'] = 'native.bash' in model_calls['advertised_tools']
                except (ValueError, IndexError):
                    pass
                try:
                    result = original_generate(request)
                except Exception as exc:
                    category = provider_failure_category(exc)
                    provider_failures.append(category)
                    from alphonse.agent_v2.core.inference.router import _simple_loop_failure_detail
                    event = {'event': 'model_failed', 'category': category}
                    detail = _simple_loop_failure_detail(exc)
                    if detail is not None:
                        event['detail'] = detail
                    emit(event)
                    raise
                model_calls['returned'] += 1
                value = result.json_value if isinstance(result.json_value, dict) else {}
                decisions.append({'kind': value.get('kind'), 'tool_id': value.get('tool_id')})
                event = {'event': 'model_decision', 'kind': value.get('kind'),
                         'tool_id': value.get('tool_id'), 'arguments': value.get('arguments'),
                         'usage': result.usage}
                # Only an explicitly synthetic scenario may retain generated
                # text, for example to review an undelivered clarification.
                if scenario is not None and getattr(scenario, 'record_decision_text', False):
                    event['text'] = value.get('text')
                emit(event)
                return result
            router.generate_json = observe_model
            web_store = None
            if synthetic_web_base_url is not None:
                from alphonse.agent_v2.web_tools_settings import SQLiteWebToolsSettingsStore, WebToolsSettings
                web_store = SQLiteWebToolsSettingsStore(path)
                web_store.save(WebToolsSettings(enabled=True, searxng_base_url=synthetic_web_base_url))
            runtime = build_runtime_host(user_store=users, messages=queue,
                question_store=SQLiteQuestionStore(path), conversation_store=conversation,
                outbox=SQLiteOutboundStore(path), tools=tools, inference=router,
                web_tools_settings_store=web_store,
                coding_skill_service=None if full_registry else object(), experimental_owner_read=True,
                owner_selected_draft_sink=lambda *_: None)
            if synthetic_clarification_review:
                class SyntheticClarificationReview:
                    def review_clarification_effect_claim(self, *, message, clarification):
                        emit({'event': 'synthetic_clarification_review', 'decision': 'allow',
                              'warning': 'not a live Jev judgment'})
                        return True
                runtime.core.system_one = SyntheticClarificationReview()
            registry = runtime.core.tools
            enabled_tools = sorted(item.tool_id for item in registry.list())
            registered_tools = sorted(registry._definitions_by_id)
            with DojoBash(image_id=image_id, docker_socket=docker_socket) as shell:
                seed(shell)
                original_execute = shell.execute
                def observe(command, **kwargs):
                    emit({'event': 'bash_started', 'command': command})
                    result = original_execute(command, **kwargs)
                    observations.append({'command': command, **result})
                    emit({'event': 'bash_finished', **result})
                    return result
                shell.execute = observe
                runtime.processor.owner_read_candidate.enable_dojo_bash(shell)
                desktop_authority = None
                if synthetic_authority:
                    # These identities exist only in the test process. This
                    # exercises wiring, not OS identity or secure ingress.
                    from alphonse.agent_v2.authority_service.prototype import (
                        AuthenticatedPeer, AuthorityDenied, AuthorityService,
                    )
                    from alphonse.agent_v2.evaluation.dojo_owner_authority import DojoOwnerAuthority
                    desktop, worker = object(), object()
                    def authenticate(connection):
                        if connection is desktop:
                            return AuthenticatedPeer('desktop', owner.user_id, 'dojo-desktop')
                        if connection is worker:
                            return AuthenticatedPeer('worker', owner.user_id, 'dojo-worker')
                        raise AuthorityDenied('unknown_dojo_peer')
                    service = AuthorityService(root / 'authority.sqlite', authenticate=authenticate)
                    # Disposable Dojo fixture explicitly opts in; production defaults off.
                    service.grant_bash(desktop, expected_generation=service.bash_capability(desktop)['generation'])
                    if deny_authority_dispatch:
                        def deny_start(*_args, **_kwargs):
                            raise AuthorityDenied('dojo_dispatch_denied')
                        service.start_action = deny_start
                    def observe_authority(event):
                        authority_events.append(event)
                        emit(event)
                    adapter = DojoOwnerAuthority(service, desktop, worker, emit=observe_authority)
                    runtime.processor.owner_read_candidate.enable_dojo_authority(adapter)
                elif socket_authority:
                    from alphonse.agent_v2.authority_service.experimental_ingress import queue_request_binding
                    from alphonse.agent_v2.authority_service.prototype import AuthorityDenied
                    from alphonse.agent_v2.authority_service.transport import AuthorityClient, AuthoritySocketService, EndpointIdentity
                    from alphonse.agent_v2.core.intelligence.simple_loop.owner_authority_bridge import OwnerAuthorityBridge
                    from alphonse.agent_v2.evaluation.dojo_owner_authority import DojoOwnerAuthority
                    # Deliberately fake verifier: this tests protocol and loop
                    # wiring, not a signed Desktop or protected service UID.
                    service = AuthoritySocketService(root / 'authority.sqlite', endpoints={
                        'desktop': EndpointIdentity('desktop', owner.user_id, 'dojo-desktop'),
                        'worker': EndpointIdentity('worker', owner.user_id, 'dojo-worker'),
                    }, verify_peer=lambda *_: True)
                    if deny_authority_dispatch:
                        def deny_start(*_args, **_kwargs):
                            raise AuthorityDenied('dojo_dispatch_denied')
                        service._ledger.start_action = deny_start
                    def client_for(role):
                        def connect():
                            client_socket, server_socket = socket.socketpair()
                            client_socket.settimeout(3)
                            threading.Thread(target=service.serve_connection, args=(server_socket,),
                                kwargs={'endpoint_role': role}, daemon=True).start()
                            return client_socket
                        return AuthorityClient(connect)
                    desktop_authority, worker_authority = client_for('desktop'), client_for('worker')
                    # Disposable Dojo fixture explicitly opts in; this is not a household grant.
                    desktop_authority.grant_bash(expected_generation=desktop_authority.bash_capability()['generation'])
                    def observe_authority(event):
                        authority_events.append(event)
                        emit(event)
                    bridge = OwnerAuthorityBridge(worker_authority, lambda task: queue_request_binding(queue, task))
                    adapter = DojoOwnerAuthority.from_worker_bridge(bridge, emit=observe_authority)
                    runtime.processor.owner_read_candidate.enable_dojo_authority(adapter)
                daemon = V2Daemon(runtime, daemon_id='owner-dojo-shopping', experimental_owner_only=True)
                daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock',
                                            experimental_authority_ingress=socket_authority)
                status, draft, acknowledged, pickup_error = 'not_started', None, False, None
                submitted_at = acknowledged_at = None
                after_owner = None
                try:
                    daemon.start()
                    client = V2DaemonClient(root / 'owner.sock')
                    identity = client.runtime_identity()
                    emit({'event': 'runtime_identity', 'loop': identity.get('loop'),
                          'loaded_code_sha256': identity.get('loaded_code_sha256'),
                          'identity_scope': identity.get('identity_scope')})
                    submitted_at = datetime.now(timezone.utc).isoformat()
                    if socket_authority:
                        from alphonse.agent_v2.authority_service.experimental_ingress import submit_owner_request
                        message_id = uuid4().hex
                        submit_owner_request(desktop_authority, client, prompt=goal, message_id=message_id)
                        emit({'event': 'synthetic_service_first_ingress'})
                    else:
                        message_id = client.request('experimental_owner_ingress', prompt=goal)['message_id']
                    deadline = time.monotonic() + timeout_seconds
                    payload = {'status': 'pending'}
                    while time.monotonic() < deadline:
                        payload = client.request('experimental_owner_status', message_id=message_id)
                        status = payload['status']
                        if status not in {'pending', 'processing'}:
                            break
                        time.sleep(.02)
                    if status == 'ready':
                        try:
                            draft = client.request('experimental_owner_take_draft', message_id=message_id)
                            client.request('experimental_owner_ack_draft', message_id=message_id,
                                acknowledgement_token=draft['acknowledgement_token'])
                            acknowledged = client.request('experimental_owner_status', message_id=message_id)['status'] == 'client_acknowledged'
                            if acknowledged:
                                acknowledged_at = datetime.now(timezone.utc).isoformat()
                        except (ValueError, RuntimeError, OSError) as exc:
                            pickup_error = type(exc).__name__
                            emit({'event': 'pickup_failed', 'error_type': pickup_error})
                    history = conversation.list(owner_user_id=owner.user_id)
                    history_recorded = bool(draft and history and history[-1].content == draft.get('message'))
                    if scenario is not None and hasattr(scenario, 'after_owner'):
                        prior_bash, prior_decisions = len(observations), len(decisions)
                        after_owner = scenario.after_owner(daemon, runtime, owner.user_id, emit, client=client)
                        if after_owner is not None:
                            after_owner['tool_activity_during_handoff'] = {
                                'model_decisions': decisions[prior_decisions:],
                                'bash_calls': observations[prior_bash:]}
                finally:
                    daemon.stop()
                shell.execute = original_execute
                try:
                    readback = readback_effect(shell)
                except Exception:
                    readback = {'status': 'unavailable'}
                if scenario is not None and hasattr(scenario, 'readback_runtime'):
                    readback['runtime'] = scenario.readback_runtime(path, owner.user_id, started_at, runtime)
                if after_owner is not None:
                    readback['after_owner'] = after_owner
                readback['submission_window'] = {'submitted_at': submitted_at, 'acknowledged_at': acknowledged_at}
            scored = grade_effect(readback)
            assessment = _capability_outcome(status, payload.get('failure_code'), decisions,
                                             observations, scored['effect_passed'], provider_failures,
                                             authority_events)
            return {'case': case_id, 'status': status,
                    'failure_code': payload.get('failure_code'),
                    'provider_failures': provider_failures,
                    'provider_calls': model_calls['started'], 'provider_calls_returned': model_calls['returned'],
                    'bash_advertised': model_calls['bash_advertised'], 'inference': 'openai_codex' if model else 'scripted',
                    'registry_mode': 'default_synthetic' if full_registry else 'reduced_fixture',
                    'registered_tools': registered_tools, 'enabled_tools': enabled_tools,
                    'synthetic_authority': synthetic_authority or socket_authority,
                    'socket_authority': socket_authority, 'authority_events': authority_events,
                    'synthetic_clarification_review': synthetic_clarification_review,
                    'advertised_tools': model_calls['advertised_tools'],
                    'bash_calls': len(observations), 'observed_readback': readback, **scored,
                    'answer': draft.get('message') if draft else None,
                    'pickup_error_type': pickup_error,
                    'client_acknowledged': acknowledged, 'history_recorded': history_recorded,
                    'cleanup_verified': shell.cleanup_verified,
                    'elapsed_seconds': round(time.monotonic() - started, 3),
                    **assessment,
                    'limitations': ["One synthetic development attempt is not a reliability estimate." if model else
                                    'Scripted model decisions; not an autonomous result.',
                        'Owner worker uses injected container Bash, not household host Bash.',
                        'Service-first socket protocol uses fake identities and does not establish an OS security boundary.' if socket_authority else
                        'Synthetic authority uses in-process fake identities and does not establish an OS security boundary.' if synthetic_authority else
                        'No synthetic authority service is connected to this run.',
                        'Clarification effect-claim review was stubbed to allow; this does not test Jev or claim safety.' if synthetic_clarification_review else
                        'Clarification review was not stubbed by this harness.',
                        'The separate inference provider uses its own environment and may have unobserved tool activity.',
                        'Result readback shares a trusted container; response semantics are unassessed.',
                        'A client acknowledgement is not proof of human delivery.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', help='explicitly enable the configured Codex model')
    parser.add_argument('--full-registry', action='store_true', help='build the default synthetic native registry')
    parser.add_argument('--synthetic-authority', action='store_true', help='exercise the in-process Dojo authority prototype')
    parser.add_argument('--socket-authority', action='store_true', help='exercise service-first synthetic socket authority')
    parser.add_argument('--deny-authority-dispatch', action='store_true', help='synthetically deny action dispatch')
    parser.add_argument('--synthetic-clarification-review', action='store_true', help='stub clarification effect-claim review; not a Jev test')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(row):
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': 'owner-shopping', 'image_id': args.image_id,
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_tree_sha256': digest.hexdigest(), 'inference': 'openai_codex' if args.model else 'scripted',
              'model': args.model,
              'authority_mode': ('service_first_socketpair_fake_peers' if args.socket_authority else
                                 'in_process_fake_peers' if args.synthetic_authority else 'none'),
              'scope': 'owner daemon IPC and disposable container; no installed Desktop or household data'})
        directory_fd = os.open(args.journal.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                         model=args.model, timeout_seconds=120 if args.model else 15,
                         full_registry=args.full_registry, synthetic_authority=args.synthetic_authority,
                         deny_authority_dispatch=args.deny_authority_dispatch,
                         synthetic_clarification_review=args.synthetic_clarification_review,
                         socket_authority=args.socket_authority)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error',
                  'task_passed': None, 'error_type': type(exc).__name__})
            raise
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
