from __future__ import annotations

import os
from typing import Any

from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.providers.opencode import OpenCodeClient
from alphonse.agent.cognition.providers.openai import OpenAIClient


def build_llm_client() -> Any:
    provider = str(os.getenv("ALPHONSE_LLM_PROVIDER", "ollama")).strip().lower()
    if provider == "openai":
        return _build_openai_client()
    if provider == "opencode":
        return _build_opencode_client()
    return _build_ollama_client()


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
        "ollama/mistral:7b-instruct",
    )
    timeout_seconds = _parse_float(os.getenv("OPENCODE_TIMEOUT_SECONDS"), default=120.0)
    chat_path = os.getenv("OPENCODE_CHAT_COMPLETIONS_PATH", "/v1/chat/completions")
    return OpenCodeClient(
        base_url=base_url,
        model=model,
        timeout=timeout_seconds,
        chat_path=chat_path,
        api_key_env=os.getenv("OPENCODE_API_KEY_ENV", "OPENCODE_API_KEY"),
        username_env=os.getenv("OPENCODE_USERNAME_ENV", "OPENCODE_SERVER_USERNAME"),
        password_env=os.getenv("OPENCODE_PASSWORD_ENV", "OPENCODE_SERVER_PASSWORD"),
    )


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
