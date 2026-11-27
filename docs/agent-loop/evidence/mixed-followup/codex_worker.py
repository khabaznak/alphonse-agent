"""Distinct Codex CLI worker and independent verifier for the synthetic preview Dojo."""

import ast
from dataclasses import asdict
import difflib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from alphonse.agent_v2.core.inference.openai_codex import _reported_codex_usage
from alphonse.agent_v2.core.intelligence.simple_loop.owner_disposable_handoff import DisposableHandoffReceipt


_TRACKED = ('preview.py', 'check_preview.py')
_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['reproduces', 'path', 'replacement'],
    'properties': {
        'reproduces': {'type': 'boolean'}, 'path': {'type': 'string', 'enum': ['preview.py']},
        'replacement': {'type': 'string'},
    },
}
_BRIEF = ('Review this synthetic, pinned repository snapshot as a separate Codex worker. The complaint is that '
          'print preview cuts off long subtitles. Determine whether the behavior in preview.py reproduces the '
          'complaint by inspecting the supplied files. If so, propose the smallest correction, and return a '
          'JSON object with reproduces=true, path="preview.py", and the complete replacement file as replacement. '
          'If the complaint does not reproduce, set reproduces=false and use an empty replacement. Preserve the '
          'existing subtitle function and its public signature. Use only the supplied snapshots. '
          'Return the proposal only. Do not call tools, execute commands, inspect other files, or apply anything.')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write_record(path, record):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def _fixture_contract(proposal):
    """Admit the one-file, no-effects form before independent execution."""
    if (type(proposal) is not dict or set(proposal) != {'reproduces', 'path', 'replacement'}
            or proposal['reproduces'] is not True or proposal['path'] != 'preview.py'
            or type(proposal['replacement']) is not str or len(proposal['replacement'].encode()) > 512):
        return False
    try:
        observed = ast.parse(proposal['replacement'].encode('utf-8'))
    except (SyntaxError, ValueError):
        return False
    safe = ast.parse('def render_subtitle(text):\n    return text\n')
    return ast.dump(observed) == ast.dump(safe)


class CodexProposalReceiver:
    """One receipt, one CLI process, one bounded proposal, and separate readback."""

    def __init__(self, root, *, repo, target, revision, model, cli):
        self.root = Path(root)
        self.root.mkdir()
        self.repo = Path(repo)
        self.target, self.revision, self.model = target, revision, model
        self.cli = Path(cli).resolve(strict=True)
        # Launch the native executable directly so the OS can deny all other
        # process executables, including interpreters and shells.
        with self.cli.open('rb') as executable:
            if sys.platform != 'darwin' or executable.read(4) not in {b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe'}:
                raise ValueError('fixture_requires_native_macos_codex')
        self.evidence = {'worker_model': self.model, 'cli_starts': [], 'verifications': []}

    def _git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.DEVNULL, timeout=5)

    def _snapshots(self):
        if (self._git('rev-parse', 'HEAD').decode().strip() != self.revision
                or set(self._git('ls-tree', '-r', '--name-only', self.revision).decode().splitlines()) != set(_TRACKED)):
            raise ValueError('fixture_revision_changed')
        files = {name: self._git('show', f'{self.revision}:{name}') for name in _TRACKED}
        if any(len(data) > 4096 or (self.repo / name).is_symlink() or (self.repo / name).read_bytes() != data
               for name, data in files.items()):
            raise ValueError('fixture_snapshot_changed')
        return files

    def _paths(self, request):
        prefix = _sha(request.operation_id.encode())
        return self.root / (prefix + '.accept.json'), self.root / (prefix + '.result.json')

    def receipts(self):
        return [json.loads(path.read_text(encoding='utf-8')) for path in sorted(self.root.glob('*.accept.json'))]

    def _proposal(self, files, claim_active):
        prompt = _BRIEF + '\n\n' + json.dumps({name: data.decode('utf-8') for name, data in files.items()})
        evidence = {'prompt_sha256': _sha(prompt.encode()), 'snapshot_sha256': {name: _sha(data) for name, data in files.items()},
                    'elapsed_seconds': None, 'tool_item_types': [], 'usage': {}}
        start = time.monotonic()
        temp_parent = self.root.resolve()
        with TemporaryDirectory(prefix='cli-', dir=temp_parent) as folder:
            workspace = Path(folder)
            schema = workspace / 'schema.json'
            schema.write_text(json.dumps(_SCHEMA), encoding='utf-8')
            final = workspace / 'answer.json'
            env = {name: os.environ[name] for name in ('HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'PATH', 'CODEX_HOME')
                   if name in os.environ}
            env['TMPDIR'] = folder
            home = Path(env['HOME']).resolve()
            codex_state = Path(env.get('CODEX_HOME', home / '.codex')).resolve()

            def subpath(path):
                return '(subpath ' + json.dumps(str(path)) + ')'

            blocked = ' '.join(subpath(codex_state / name) for name in ('skills', 'sessions', 'archived_sessions'))
            policy = ('(version 1) (allow default) '
                      '(deny process-exec (require-not (literal ' + json.dumps(str(self.cli)) + '))) '
                      '(deny file-read-data file-write* (require-all ' + subpath(home)
                      + ' (require-not ' + subpath(codex_state) + ') (require-not ' + subpath(self.cli.parent) + '))) '
                      '(deny file-read* file-write* ' + blocked + ' ' + subpath(self.repo.resolve()) + ')')
            command = ['/usr/bin/sandbox-exec', '-p', policy, str(self.cli),
                       '-c', 'model_provider="openai"', '-c', 'cli_auth_credentials_store="auto"',
                       '-c', 'web_search="disabled"', '-c', 'mcp_servers={}', '--ask-for-approval', 'never',
                       'exec', '--skip-git-repo-check', '--ignore-user-config', '--ignore-rules', '--ephemeral',
                       '--sandbox', 'read-only', '--model', self.model,
                       '--json', '--output-schema', str(schema), '--output-last-message', str(final)]
            for feature in ('shell_tool', 'unified_exec', 'apps', 'browser_use', 'in_app_browser',
                            'tool_suggest', 'multi_agent', 'multi_agent_v2'):
                command.extend(('--disable', feature))
            command.append('-')
            if claim_active() is not True:
                raise ValueError('fixture_worker_claim_lost')
            with subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, encoding='utf-8', start_new_session=True) as process:
                evidence['pid'] = process.pid
                evidence['execution_boundary'] = 'macOS executable allowlist, home and Git fixture access restrictions'
                self.evidence['cli_starts'].append(evidence)
                pending_input = prompt
                try:
                    while True:
                        if claim_active() is not True or time.monotonic() - start > 50:
                            raise ValueError('fixture_worker_stopped')
                        try:
                            output, diagnostics = process.communicate(pending_input, timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            pending_input = None
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                    evidence['elapsed_seconds'] = round(time.monotonic() - start, 3)
                evidence['exit_code'] = process.returncode
                if process.returncode:
                    diagnostic = diagnostics.lower()
                    evidence['exit_category'] = next((category for marker, category in (
                        ('unexpected argument', 'cli_arguments_rejected'),
                        ('sandbox-exec:', 'sandbox_start_failed'),
                        ('failed to load', 'cli_configuration_failed'),
                        ('unsupported', 'cli_or_model_unsupported'),
                    ) if marker in diagnostic), 'cli_failure')
            if len(output.encode()) > 131072:
                raise ValueError('fixture_worker_output_too_large')
            events = [json.loads(line) for line in output.splitlines() if line.strip()]
            evidence['thread_started'] = sum(event.get('type') == 'thread.started' for event in events)
            evidence['turn_completed'] = sum(event.get('type') == 'turn.completed' for event in events)
            evidence['usage'] = _reported_codex_usage(output)
            evidence['tool_item_types'] = sorted({item['type'] for event in events
                if isinstance(item := event.get('item'), dict) and isinstance(item.get('type'), str)
                and item['type'] not in {'agent_message', 'reasoning'}})
            if (process.returncode != 0 or evidence['thread_started'] != 1 or evidence['turn_completed'] != 1
                    or evidence['tool_item_types'] or not final.is_file() or final.stat().st_size > 4096):
                raise ValueError('fixture_worker_response_unverified')
            return json.loads(final.read_text(encoding='utf-8'))

    def submit(self, request, claim_active):
        if (claim_active() is not True or (request.target, request.revision, request.worker, request.operation)
                != (self.target, self.revision, 'codex', 'propose_patch')):
            return None
        files = self._snapshots()
        accepted, result = self._paths(request)
        _write_record(accepted, {'request': asdict(request), 'status': 'accepted'})
        try:
            proposal = self._proposal(files, claim_active)
        except Exception as exc:
            self.evidence['worker_error'] = str(exc) if type(exc) is ValueError and str(exc).startswith('fixture_') else type(exc).__name__
            return DisposableHandoffReceipt(request, 'failed')
        _write_record(result, {'request': asdict(request), 'proposal': proposal})
        return DisposableHandoffReceipt(request, 'completed')

    def reconcile(self, request, claim_active):
        accepted, result = self._paths(request)
        if claim_active() is not True or not accepted.is_file():
            return None
        if json.loads(accepted.read_text(encoding='utf-8')) != {'request': asdict(request), 'status': 'accepted'}:
            return None
        return DisposableHandoffReceipt(request, 'completed' if result.is_file() else 'accepted')

    def verify_result(self, request, receipt, claim_active):
        if claim_active() is not True or receipt.request != request:
            return False
        _, result = self._paths(request)
        record = json.loads(result.read_text(encoding='utf-8'))
        proposal = record.get('proposal')
        evidence = {'operation_id': request.operation_id, 'safe_single_file_proposal': _fixture_contract(proposal)}
        self.evidence['verifications'].append(evidence)
        if record.get('request') != asdict(request) or not evidence['safe_single_file_proposal']:
            return False
        original = self._snapshots()
        candidate = {**original, 'preview.py': proposal['replacement'].encode()}
        evidence['proposal_diff'] = ''.join(difflib.unified_diff(
            original['preview.py'].decode().splitlines(keepends=True), proposal['replacement'].splitlines(keepends=True),
            fromfile='a/preview.py', tofile='b/preview.py'))
        with TemporaryDirectory(prefix='verify-', dir=self.root) as folder:
            directory = Path(folder)
            for files, name in ((original, 'baseline'), (candidate, 'candidate')):
                copy = directory / name
                copy.mkdir()
                for filename, data in files.items():
                    (copy / filename).write_bytes(data)
                check = subprocess.run([sys.executable, '-E', '-S', '-B', str(copy / 'check_preview.py')],
                                       cwd=copy, env={}, capture_output=True, timeout=5, check=False)
                evidence[f'{name}_test_exit'] = check.returncode
                if name == 'baseline':
                    evidence['baseline_assertion_failed'] = b'AssertionError' in check.stderr
        evidence['pinned_original_unchanged'] = self._snapshots() == original
        return (claim_active() is True and evidence['baseline_test_exit'] == 1 and evidence['baseline_assertion_failed']
                and evidence['candidate_test_exit'] == 0 and evidence['pinned_original_unchanged'])
