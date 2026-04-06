from __future__ import annotations

import os
from typing import Any

from alphonse.agent.cognition.providers.contracts import TextCompletionProvider
from alphonse.agent.cognition.providers.contracts import ToolCallingProvider
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.providers.contracts import require_tool_calling_provider
from alphonse.agent.cognition.providers.llamafarm import LlamaFarmClient
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.providers.opencode import OpenCodeClient
from alphonse.agent.cognition.providers.openai import OpenAIClient

_CLIENT_CACHE: dict[tuple[str, tuple[str | None, ...]], Any] = {}


def build_llm_client() -> TextCompletionProvider:
    return build_text_completion_provider()


def build_text_completion_provider() -> TextCompletionProvider:
    provider = str(os.getenv("ALPHONSE_LLM_PROVIDER", "ollama")).strip().lower()
    client = _build_cached_provider(provider)
    return require_text_completion_provider(client, source=f"providers.factory:{provider or 'default'}")


def build_tool_calling_provider() -> ToolCallingProvider:
    provider = str(os.getenv("ALPHONSE_LLM_PROVIDER", "ollama")).strip().lower()
    client = _build_cached_provider(provider)
    return require_tool_calling_provider(client, source=f"providers.factory:{provider or 'default'}")


def _build_cached_provider(provider: str) -> Any:
    cache_key = (provider or "default", _provider_config_key(provider))
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = _build_provider(provider)
    return _CLIENT_CACHE[cache_key]


def _build_provider(provider: str) -> Any:
    if provider == "openai":
        return _build_openai_client()
    if provider == "opencode":
        return _build_opencode_client()
    if provider in {"llamafarm", "llama_farm"}:
        return _build_llamafarm_client()
    return _build_ollama_client()


def _provider_config_key(provider: str) -> tuple[str | None, ...]:
    if provider == "openai":
        names = (
            "OPENAI_BASE_URL",
            "OPENAI_MODEL",
            "OPENAI_TIMEOUT_SECONDS",
            "OPENAI_API_KEY_ENV",
        )
    elif provider == "opencode":
        names = (
            "OPENCODE_BASE_URL",
            "OPENCODE_MODEL",
            "LOCAL_LLM_MODEL",
            "OPENCODE_TIMEOUT_SECONDS",
            "OPENCODE_API_KEY_ENV",
            "OPENCODE_USERNAME_ENV",
            "OPENCODE_SERVER_USERNAME",
            "OPENCODE_PASSWORD_ENV",
            "OPENCODE_SERVER_PASSWORD",
        )
    elif provider in {"llamafarm", "llama_farm"}:
        names = (
            "LLAMAFARM_BASE_URL",
            "LLAMAFARM_MODEL",
            "LOCAL_LLM_MODEL",
            "LLAMAFARM_TIMEOUT_SECONDS",
            "LLAMAFARM_API_KEY_ENV",
        )
    else:
        names = (
            "OLLAMA_BASE_URL",
            "LOCAL_LLM_MODEL",
            "LOCAL_LLM_TIMEOUT_SECONDS",
        )
    return tuple(os.getenv(name) for name in names)


def _build_ollama_client() -> OllamaClient:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct")
    timeout_seconds = _parse_float(
        os.getenv("LOCAL_LLM_TIMEOUT_SECONDS"),
        default=240.0,
    )
    return OllamaClient(
        base_url=base_url,
        model=model,
        timeout=timeout_seconds,
    )


def _build_openai_client() -> OpenAIClient:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    timeout_seconds = _parse_float(os.getenv("OPENAI_TIMEOUT_SECONDS"), default=60.0)
    return OpenAIClient(
        base_url=base_url,
        model=model,
        api_key_env=os.getenv("OPENAI_API_KEY_ENV", "OPENAI_API_KEY"),
        timeout=timeout_seconds,
    )


def _build_opencode_client() -> OpenCodeClient:
    base_url = os.getenv("OPENCODE_BASE_URL", "http://127.0.0.1:4096")
    model = os.getenv("OPENCODE_MODEL") or os.getenv(
        "LOCAL_LLM_MODEL",
        "openai/gpt-5.1-codex",
    )
    timeout_seconds = _parse_float(os.getenv("OPENCODE_TIMEOUT_SECONDS"), default=60.0)
    return OpenCodeClient(
        base_url=base_url,
        model=model,
        timeout=timeout_seconds,
        api_key_env=os.getenv("OPENCODE_API_KEY_ENV", "OPENCODE_API_KEY"),
        username_env=os.getenv("OPENCODE_USERNAME_ENV", "OPENCODE_SERVER_USERNAME"),
        password_env=os.getenv("OPENCODE_PASSWORD_ENV", "OPENCODE_SERVER_PASSWORD"),
    )


def _build_llamafarm_client() -> LlamaFarmClient:
    base_url = os.getenv("LLAMAFARM_BASE_URL", "http://127.0.0.1:8002/v1")
    model = os.getenv("LLAMAFARM_MODEL") or os.getenv(
        "LOCAL_LLM_MODEL",
        "mistral:7b-instruct",
    )
    timeout_seconds = _parse_float(os.getenv("LLAMAFARM_TIMEOUT_SECONDS"), default=120.0)
    return LlamaFarmClient(
        base_url=base_url,
        model=model,
        api_key_env=os.getenv("LLAMAFARM_API_KEY_ENV", "LLAMAFARM_API_KEY"),
        timeout=timeout_seconds,
    )


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
