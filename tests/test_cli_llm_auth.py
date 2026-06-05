from __future__ import annotations

import argparse
import json

import pytest

from alphonse.agent import cli


def test_llm_auth_list_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ALPHONSE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gh-secret")

    cli._command_llm_auth(argparse.Namespace(llm_auth_command="list"))

    output = capsys.readouterr().out
    assert "openai active=true" in output
    assert "OPENAI_API_KEY=<set>" in output
    assert "sk-secret" not in output
    assert "gh-secret" not in output


def test_llm_auth_select_prints_provider_env(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._command_llm_auth(
        argparse.Namespace(llm_auth_command="select", provider="openai_codex")
    )

    output = capsys.readouterr().out
    assert "ALPHONSE_LLM_PROVIDER=openai_codex" in output
    assert "OPENAI_CODEX_CLI_BIN=codex" in output
    assert "codex login --device-auth" in output


def test_llm_auth_smoke_success_uses_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeProvider:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            assert "smoke test" in system_prompt
            assert user_prompt == "ping"
            assert cli.os.getenv("ALPHONSE_LLM_PROVIDER") == "ollama"
            return "ok"

    monkeypatch.setattr(
        "alphonse.agent.cognition.providers.factory.build_text_completion_provider",
        lambda: _FakeProvider(),
    )

    cli._command_llm_auth(
        argparse.Namespace(
            llm_auth_command="smoke",
            provider="ollama",
            text="ping",
            timeout_seconds=None,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "ollama"
    assert payload["status"] == "ok"
    assert payload["output_preview"] == "ok"


def test_llm_auth_smoke_failure_reports_structured_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeProvider:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            raise ValueError("openai_codex_auth_required")

    monkeypatch.setattr(
        "alphonse.agent.cognition.providers.factory.build_text_completion_provider",
        lambda: _FakeProvider(),
    )

    cli._command_llm_auth(
        argparse.Namespace(
            llm_auth_command="smoke",
            provider="openai_codex",
            text="ping",
            timeout_seconds=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "openai_codex"
    assert payload["status"] == "error"
    assert payload["code"] == "openai_codex_auth_required"


def test_parser_accepts_llm_auth_commands() -> None:
    args = cli.build_parser().parse_args(["llm-auth", "smoke", "--provider", "openai"])
    assert args.command == "llm-auth"
    assert args.llm_auth_command == "smoke"
    assert args.provider == "openai"
