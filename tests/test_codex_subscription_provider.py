"""Offline checks; these tests never run Codex or a model."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest, ModelProfile
from alphonse.agent_v2.evaluation import codex_subscription_provider as subject


def request(*, cancel_checker=None, model='gpt-6-luna'):
    return InferenceRequest(prompt='synthetic', purpose=InferencePurpose.TACTICAL_ACTION,
                            model_profile=ModelProfile(provider='openai_codex', model=model,
                                                       profile_id='test'),
                            cancel_checker=cancel_checker)


class FakeProcess:
    returncode = 0

    def communicate(self, timeout=None):
        return ('{"kind":"answer","text":"ok"}', '')


def test_explicit_opt_in_and_chatgpt_preflight_scrub_environment(monkeypatch):
    monkeypatch.setenv('HOME', '/synthetic/home')
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-secret')
    monkeypatch.setenv('CODEX_API_KEY', 'synthetic-secret')
    monkeypatch.setenv('CODEX_ACCESS_TOKEN', 'synthetic-secret')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://bad.invalid')
    monkeypatch.setattr(subject.shutil, 'which', lambda *args, **kwargs: '/usr/local/bin/codex')
    calls = []
    def status(command, **kwargs):
        calls.append(('status', command, kwargs))
        return SimpleNamespace(returncode=0, stdout='', stderr='Logged in using ChatGPT\n')
    def dispatch(command, **kwargs):
        calls.append(('dispatch', command, kwargs))
        return FakeProcess()
    monkeypatch.setattr(subject.subprocess, 'run', status)
    monkeypatch.setattr(subject.subprocess, 'Popen', dispatch)
    provider = subject.CodexSubscriptionProvider(model='gpt-6-luna', enabled=True)
    result = provider.generate_json(request())
    assert result.json_value == {'kind': 'answer', 'text': 'ok'}
    assert [call[0] for call in calls] == ['status', 'dispatch']
    status_env, exec_env = calls[0][2]['env'], calls[1][2]['env']
    assert status_env == exec_env
    assert calls[0][2]['cwd'] == calls[1][2]['cwd']
    assert all(name not in exec_env for name in ('OPENAI_API_KEY', 'CODEX_API_KEY',
                                                  'CODEX_ACCESS_TOKEN', 'OPENAI_BASE_URL'))
    # Execution must avoid starting the user's configured MCP servers. The
    # installed login-status command cannot ignore user config, so the shared
    # auth-critical overrides do not prove full config equivalence.
    assert '--ignore-user-config' in calls[1][1]
    for override in ('model_provider="openai"', 'cli_auth_credentials_store="auto"'):
        assert override in calls[0][1] and override in calls[1][1]
    assert '--ephemeral' in calls[1][1]
    assert 'synthetic' not in calls[1][1]  # prompt goes through stdin
    assert result.usage == {}  # never invent provider-reported usage


@pytest.mark.parametrize('status', ['Logged in using an API key', '', 'unexpected',
                                   'Logged in using ChatGPT\nLogged in using an API key'])
def test_non_chatgpt_status_fails_before_model_dispatch(monkeypatch, status):
    monkeypatch.setenv('HOME', '/synthetic/home')
    monkeypatch.setattr(subject.shutil, 'which', lambda *args, **kwargs: '/usr/local/bin/codex')
    monkeypatch.setattr(subject.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=status, stderr=''))
    monkeypatch.setattr(subject.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('model dispatch'))
    with pytest.raises(ValueError, match='subscription_chatgpt_login_required'):
        subject.CodexSubscriptionProvider(model='gpt-6-luna', enabled=True).generate_json(request())


def test_disabled_and_cancelled_stop_before_preflight(monkeypatch):
    monkeypatch.setattr(subject.subprocess, 'run', lambda *args, **kwargs: pytest.fail('preflight'))
    with pytest.raises(ValueError, match='subscription_dispatch_disabled'):
        subject.CodexSubscriptionProvider(model='gpt-6-luna').generate_json(request())
    with pytest.raises(ValueError, match='subscription_inference_cancelled'):
        subject.CodexSubscriptionProvider(model='gpt-6-luna', enabled=True).generate_json(
            request(cancel_checker=lambda: True))


def test_explicit_model_mismatch_stops_before_model_dispatch(monkeypatch):
    monkeypatch.setenv('HOME', '/synthetic/home')
    monkeypatch.setattr(subject.shutil, 'which', lambda *args, **kwargs: '/usr/local/bin/codex')
    monkeypatch.setattr(subject.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='Logged in using ChatGPT\n', stderr=''))
    monkeypatch.setattr(subject.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('model dispatch'))
    with pytest.raises(ValueError, match='subscription_model_mismatch'):
        subject.CodexSubscriptionProvider(model='gpt-6-luna', enabled=True).generate_json(request(model='other'))
