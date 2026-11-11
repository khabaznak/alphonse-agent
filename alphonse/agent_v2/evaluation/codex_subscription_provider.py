"""Evaluation-only Codex CLI provider with a ChatGPT sign-in preflight.

No process is started at import or construction. This adapter does not itself
authorize live work, prove billing mode throughout a call, or sandbox the CLI.
    It strips ambient API credentials and checks the same temporary working
    directory, child environment and auth-critical CLI overrides before dispatch.
    Execution ignores user configuration to avoid unrelated configured MCPs;
    login status on the installed CLI cannot use that flag, so full config
    equivalence is unverified. An auth/config change between check and execution
    remains a race; this is not a hard guarantee against API billing.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider, _terminate_process_group


_CHILD_VARIABLES = ('HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL')
_SEARCH_PATH = '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin'
_PROVIDER_OVERRIDE = 'model_provider="openai"'
_CREDENTIAL_STORE_OVERRIDE = 'cli_auth_credentials_store="auto"'


class CodexSubscriptionProvider(OpenAICodexProvider):
    """Use the existing Codex inference envelopes with explicit opt-in.

    Provider usage remains unavailable unless instrumented separately. The
    CLI executable and local credential store are trusted inputs. Scrubbing
    inherited variables and checking login status cannot prove isolation,
    account entitlements, or the absence of an auth change during a run.
    """

    def __init__(self, *, model: str, timeout_seconds: float = 60, enabled: bool = False):
        super().__init__()
        if not isinstance(model, str) or not model.strip() or model != model.strip():
            raise ValueError('subscription_model_required')
        if type(enabled) is not bool or isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds < float('inf'):
            raise ValueError('invalid_subscription_configuration')
        self.model, self.timeout_seconds, self.enabled = model, float(timeout_seconds), enabled

    def _run_codex(self, prompt, request):
        if not self.enabled:
            raise ValueError('subscription_dispatch_disabled')
        if request.images:
            raise ValueError('subscription_fixture_images_unsupported')
        def is_cancelled():
            try:
                return bool(request.cancel_checker and request.cancel_checker())
            except Exception:
                return True
        if is_cancelled():
            raise ValueError('subscription_inference_cancelled')
        env = {key: os.environ[key] for key in _CHILD_VARIABLES if key in os.environ}
        if not env.get('HOME'):
            raise ValueError('subscription_home_required')
        env['PATH'] = _SEARCH_PATH
        executable = shutil.which('codex', path=_SEARCH_PATH)
        if not executable:
            raise ValueError('subscription_codex_unavailable')
        options = ['-c', _PROVIDER_OVERRIDE, '-c', _CREDENTIAL_STORE_OVERRIDE]
        with tempfile.TemporaryDirectory(prefix='alphonse-subscription-eval-') as directory:
            try:
                check = subprocess.run([executable, *options, 'login', 'status'], cwd=directory,
                    env=env, capture_output=True, text=True, timeout=10, check=False)
            except (OSError, subprocess.TimeoutExpired):
                raise ValueError('subscription_auth_unverified') from None
            # Discard all other output; it can include sensitive diagnostics.
            lines = (str(check.stdout or '') + '\n' + str(check.stderr or '')).splitlines()
            auth_lines = [line.strip() for line in lines if line.strip().lower().startswith('logged in')]
            if check.returncode != 0 or auth_lines != ['Logged in using ChatGPT']:
                raise ValueError('subscription_chatgpt_login_required')
            model = getattr(getattr(request, 'model_profile', None), 'model', None)
            if model and model != self.model:
                raise ValueError('subscription_model_mismatch')
            if is_cancelled():
                raise ValueError('subscription_inference_cancelled')
            command = [executable, *options, 'exec', '--ignore-user-config', '--skip-git-repo-check',
                       '--ephemeral', '--sandbox', 'read-only', '--model', self.model]
            if request.output_schema is not None:
                import json
                schema = Path(directory) / 'response-schema.json'
                schema.write_text(json.dumps(request.output_schema), encoding='utf-8')
                command.extend(['--output-schema', str(schema)])
            try:
                with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as input_file:
                    input_file.write(prompt)
                    input_file.seek(0)
                    process = subprocess.Popen(command, stdin=input_file, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, cwd=directory, env=env, start_new_session=True)
                    started = time.monotonic()
                    while True:
                        if is_cancelled():
                            _terminate_process_group(process)
                            process.communicate()
                            raise ValueError('subscription_inference_cancelled')
                        remaining = self.timeout_seconds - (time.monotonic() - started)
                        if remaining <= 0:
                            _terminate_process_group(process)
                            process.communicate()
                            raise ValueError('subscription_inference_timeout')
                        try:
                            stdout, _stderr = process.communicate(timeout=min(0.1, remaining))
                        except subprocess.TimeoutExpired:
                            continue
                        break
            except OSError:
                raise ValueError('subscription_inference_failed') from None
            if process.returncode != 0 or not str(stdout or '').strip():
                raise ValueError('subscription_inference_failed')
            return str(stdout).strip()
