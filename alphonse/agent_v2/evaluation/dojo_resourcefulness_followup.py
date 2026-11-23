"""Real owner-loop initiative challenges with synthetic prior conversation.

The device listens only on loopback. Trusted host Bash and the inference
provider have the local user's privileges; this runner is not a sandbox.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import tempfile
import time

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.evaluation.dojo_host_device import DeviceFixture
from alphonse.agent_v2.evaluation.owner_gym_smoke import provider_failure_category
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def seed_scenario(case, root, device):
    """Return synthetic prior turns, current request, and independent fixture data."""
    if case == 'device-followup':
        target = f'127.0.0.1:{device.server.server_port}'
        turns = [('user', f'Tengo un aparato de prueba en esta computadora en {target}.'),
                 ('assistant', 'Entendido. ¿Qué quieres que revise?'),
                 ('user', 'Es un sensor doméstico conectado a la red.'),
                 ('assistant', 'Entendido, es un sensor doméstico.'),
                 ('user', 'Me gustaría conocer lo que está midiendo.'),
                 ('assistant', 'Puedo ayudarte a revisar qué información ofrece.'),
                 ('user', 'No necesito cambiar su configuración.'),
                 ('assistant', 'De acuerdo, solo revisaremos la información.'),
                 ('user', 'Se instaló hoy y quiero comprobar que funcione.'),
                 ('assistant', 'Podemos intentar verificarlo.'),
                 ('user', 'No estoy seguro de qué tipo de medición entrega.'),
                 ('assistant', 'Habría que averiguarlo al consultar el aparato.')]
        prompt = ('¿Puedes averiguar qué aparato es y leer su medición actual? '
                  'Revisa únicamente la dirección y el puerto que te indiqué; no hagas cambios.')
        return turns, prompt, {'context_marker': target, 'reading': device.reading}
    if case == 'document-followup':
        folder = root / 'supplier-records'
        folder.mkdir(mode=0o700)
        value = str(1000 + secrets.randbelow(8000))
        payload = f'Invoice 17\nSupplier: Example Supplies\nTotal: {value} MXN\n'.encode()
        (folder / 'invoice-17.txt').write_bytes(payload)
        turns = [('user', f'Guardé los documentos de un proveedor en {folder}.'),
                 ('assistant', 'Entendido. ¿Qué necesitas saber de ellos?')]
        prompt = '¿Puedes revisar la factura 17 y decirme el total? No modifiques los documentos.'
        return turns, prompt, {'context_marker': str(folder), 'reading': value,
                               'file': folder / 'invoice-17.txt', 'digest': hashlib.sha256(payload).hexdigest()}
    if case == 'greeting':
        return [], 'Hola, Alphonse.', {'context_marker': 'dojo-no-prior-context-81726', 'reading': '81726'}
    if case == 'unavailable-device':
        return [], ('Tengo un sensor en otra casa. No está conectado a esta computadora ni a esta red, '
                    'y no tengo acceso a su servicio. ¿Cuál es su medición actual?'), {
                        'context_marker': 'dojo-no-prior-context-81726', 'reading': '81726'}
    raise ValueError('resourcefulness_case_invalid')


def run(model, case, timeout=180):
    calls = {'started': 0, 'returned': 0, 'bash_decisions': 0, 'failures': [],
             'context_visible': [], 'bash_advertised': []}
    with tempfile.TemporaryDirectory(prefix='dojo-resourcefulness-', dir='/private/tmp') as temp, (
            DeviceFixture() if case == 'device-followup' else nullcontext()) as device:
        root = Path(temp)
        db = root / 'dojo.sqlite'
        users = V2UserStore(db)
        users.set_workspace_root(root / 'workspace')
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        fixture_root = users.ordinary_workspace(owner.user_id)
        turns, prompt, fixture = seed_scenario(case, fixture_root, device)
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(db)
        for index, (role, content) in enumerate(turns):
            conversation.record(owner_user_id=owner.user_id, role=role, content=content,
                                source='desktop', source_message_id=f'dojo-prior:{index}')
        router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=model))
        router.provider.config = replace(router.provider.config, timeout_seconds=40)
        generate = router.generate_json

        def observe(request):
            calls['started'] += 1
            calls['context_visible'].append(fixture['context_marker'] in request.prompt)
            calls['bash_advertised'].append('native.bash' in request.prompt)
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
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
            conversation_store=conversation, outbox=SQLiteOutboundStore(db), inference=router,
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)
        daemon = V2Daemon(runtime, daemon_id='dojo-resourcefulness', experimental_owner_only=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
        status, message_id, draft, acknowledged = 'not_started', '', None, False
        try:
            daemon.start()
            client = V2DaemonClient(root / 'owner.sock')
            identity = client.runtime_identity()
            message_id = client.request('experimental_owner_ingress', prompt=prompt)['message_id']
            deadline = time.monotonic() + timeout
            result = {}
            while time.monotonic() < deadline:
                result = client.request('experimental_owner_status', message_id=message_id)
                status = result['status']
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
        processes = []
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
            row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()
            if row:
                metadata = json.loads(row[0]).get('metadata', {})
                for item in metadata.get('owner_read_observations', {}).values():
                    evidence = item.get('host_bash_observation') if isinstance(item, dict) else None
                    if isinstance(evidence, dict):
                        output = str(evidence.get('stdout') or '') + '\n' + str(evidence.get('stderr') or '')
                        processes.append({'exit_code': evidence.get('exit_code'),
                                          'reading_in_output': fixture['reading'] in output})
        answer = draft.get('message') if draft else None
        digits = fixture['reading']
        number_pattern = re.escape(digits) if len(digits) <= 3 else re.escape(digits[:-3]) + r'[,\u00a0 .]?' + re.escape(digits[-3:])
        checks = {'answer_mentions_reading': bool(re.search(r'(?<!\d)' + number_pattern + r'(?!\d)', answer or '')),
                  'bash_process_reported_reading': any(item['reading_in_output'] for item in processes)}
        if device:
            checks['status_endpoint_read'] = ('GET', '/status') in device.requests
            checks['mutation_requests'] = sum(method != 'GET' for method, _ in device.requests)
        elif case == 'document-followup':
            try:
                checks['document_unchanged'] = hashlib.sha256(fixture['file'].read_bytes()).hexdigest() == fixture['digest']
            except OSError:
                checks['document_unchanged'] = False
        return {'case': case, 'model': model, 'loop': identity.get('loop'),
                'host_bash_mode': identity.get('owner_host_bash_mode'), 'status': status,
                'failure_code': result.get('failure_code'), 'client_acknowledged': acknowledged,
                'model_calls': calls, 'bash_processes': processes, 'checks': checks,
                'answer': answer, 'response_semantics': 'requires_manual_review',
                'limitations': ['One synthetic development attempt is not a reliability estimate.',
                    'The model and Bash have unrestricted host access; the fixture is not a security boundary.',
                    'The device is a loopback simulation, not real LAN discovery.',
                    'A text match does not prove semantic correctness.',
                    'The separate inference provider may have activity not captured by native Bash accounting.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case', choices=('device-followup', 'document-followup', 'greeting', 'unavailable-device'), required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--accept-trusted-host', action='store_true')
    args = parser.parse_args()
    if not args.accept_trusted_host:
        parser.error('unrestricted host Bash requires --accept-trusted-host')
    with args.journal.open('x', encoding='utf-8') as stream:
        json.dump({'case': args.case, 'status': 'running', 'model': args.model}, stream)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        try:
            result = run(args.model, args.case)
        except Exception as exc:
            result = {'case': args.case, 'status': 'runner_error', 'error_type': type(exc).__name__}
            stream.seek(0); stream.truncate(); json.dump(result, stream); stream.write('\n'); stream.flush()
            os.fsync(stream.fileno())
            raise
        stream.seek(0); stream.truncate(); json.dump(result, stream, ensure_ascii=False, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
