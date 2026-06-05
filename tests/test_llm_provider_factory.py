from __future__ import annotations

import pytest

from alphonse.agent.cognition.providers.factory import build_llm_client
from alphonse.agent.cognition.providers.factory import build_text_completion_provider
from alphonse.agent.cognition.providers.factory import build_tool_calling_provider
from alphonse.agent.cognition.providers.llamafarm import LlamaFarmClient
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.providers.opencode import OpenCodeClient
from alphonse.agent.cognition.providers.openai import OpenAIClient
from alphonse.agent.cognition.providers.openai_codex import OpenAICodexClient
from alphonse.agent.cognition.providers.github_copilot import GitHubCopilotClient
import alphonse.agent.cognition.providers.factory as factory_module


@pytest.mark.parametrize("provider", ["ollama", "OLLAMA", "unknown"])
def test_build_llm_client_defaults_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", provider)
    client = build_llm_client()
    assert isinstance(client, OllamaClient)


@pytest.mark.parametrize("provider", ["opencode", "OPENCODE"])
def test_build_llm_client_supports_opencode(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", provider)
    client = build_llm_client()
    assert isinstance(client, OpenCodeClient)


@pytest.mark.parametrize("provider", ["llamafarm", "llama_farm", "LLAMAFARM"])
def test_build_llm_client_supports_llamafarm(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", provider)
    client = build_llm_client()
    assert isinstance(client, LlamaFarmClient)


@pytest.mark.parametrize("provider", ["openai", "OPENAI"])
def test_build_llm_client_supports_openai(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", provider)
    client = build_llm_client()
    assert isinstance(client, OpenAIClient)


@pytest.mark.parametrize("provider", ["openai_codex", "openai-codex", "codex"])
def test_build_llm_client_supports_openai_codex(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", provider)
    client = build_llm_client()
    assert isinstance(client, OpenAICodexClient)


@pytest.mark.parametrize("provider", ["github_copilot", "github-copilot", "copilot"])
def test_build_llm_client_supports_github_copilot(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", provider)
    client = build_llm_client()
    assert isinstance(client, GitHubCopilotClient)


def test_build_llm_client_rejects_provider_without_complete_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", "unknown")
    monkeypatch.setattr(factory_module, "_build_ollama_client", lambda: object())
    factory_module._CLIENT_CACHE.clear()
    with pytest.raises(ValueError) as exc:
        build_llm_client()
    assert "provider_contract_error:text_completion_missing" in str(exc.value)


def test_build_tool_calling_provider_rejects_provider_without_tool_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", "unknown")
    monkeypatch.setattr(factory_module, "_build_ollama_client", lambda: object())
    factory_module._CLIENT_CACHE.clear()
    with pytest.raises(ValueError) as exc:
        build_tool_calling_provider()
    assert "provider_contract_error:text_completion_missing" in str(exc.value)


def test_build_text_completion_provider_reuses_cached_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", "unknown")
    class _FakeClient:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            _ = (system_prompt, user_prompt)
            return ""

    sentinel = _FakeClient()
    monkeypatch.setattr(factory_module, "_build_ollama_client", lambda: sentinel)
    factory_module._CLIENT_CACHE.clear()
    first = build_text_completion_provider()
    second = build_text_completion_provider()
    assert first is sentinel
    assert second is sentinel


def test_openai_cache_key_includes_project_header_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_a")
    factory_module._CLIENT_CACHE.clear()
    first = build_text_completion_provider()
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_b")
    second = build_text_completion_provider()
    assert first is not second


def test_codex_cache_key_includes_cli_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", "openai_codex")
    monkeypatch.setenv("OPENAI_CODEX_CLI_BIN", "codex-a")
    factory_module._CLIENT_CACHE.clear()
    first = build_text_completion_provider()
    monkeypatch.setenv("OPENAI_CODEX_CLI_BIN", "codex-b")
    second = build_text_completion_provider()
    assert first is not second
