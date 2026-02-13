from __future__ import annotations

import pytest

from alphonse.agent.cognition.providers.factory import build_llm_client
from alphonse.agent.cognition.providers.llamafarm import LlamaFarmClient
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.providers.opencode import OpenCodeClient


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
