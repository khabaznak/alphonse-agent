from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRequest
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import OpenAICodexProvider
from alphonse.agent_v2.core.inference import OpenAICodexProviderConfig


def test_codex_provider_markdown_maps_stdout_to_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout="1.- [ ] File exists\n", stderr="")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    provider = OpenAICodexProvider(OpenAICodexProviderConfig(model="fallback-model"))
    result = provider.generate_markdown(
        InferenceRequest(
            prompt="Generate acceptance criteria",
            purpose=InferencePurpose.ACCEPTANCE_CRITERIA,
            project_id="alpha",
            user="alex",
            task_id="task-1",
            model_profile=ModelProfile(provider="openai_codex", model="gpt-plus", profile_id="plus"),
        )
    )

    assert result.content == "1.- [ ] File exists"
    assert captured["command"] == ["codex", "exec", "--skip-git-repo-check", "--model", "gpt-plus"]
    assert "alphonse-codex-" in str(captured["cwd"])
    envelope = json.loads(str(captured["input"]))
    assert envelope["purpose"] == "acceptance_criteria"
    assert envelope["project_id"] == "alpha"
    assert envelope["user"] == "alex"
    assert envelope["task_id"] == "task-1"
    assert envelope["prompt"] == "Generate acceptance criteria"


def test_codex_provider_respects_explicit_default_over_environment_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="OK", stderr="")

    monkeypatch.setenv("OPENAI_CODEX_MODEL", "environment-model")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    OpenAICodexProvider().generate_markdown(
        InferenceRequest(
            prompt="Validate default",
            purpose=InferencePurpose.ACCEPTANCE_CRITERIA,
            model_profile=ModelProfile(provider="openai_codex", model="", profile_id="default"),
        )
    )

    assert "--model" not in captured["command"]


def test_codex_provider_json_parses_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout='```json\n{"ok": true}\n```', stderr="")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    result = OpenAICodexProvider().generate_json(
        InferenceRequest(prompt="Return JSON", purpose=InferencePurpose.CRITERIA_REVIEW)
    )

    assert result.json_value == {"ok": True}


def test_codex_provider_plan_tool_call_transports_tools_and_normalizes_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    planned = {
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {"path": "a.txt"},
        "internal_state": "Writing the file.",
    }

    def fake_run(command, **kwargs):
        captured["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(planned), stderr="")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    result = OpenAICodexProvider().plan_tool_call(
        InferenceRequest(
            prompt="Plan",
            purpose=InferencePurpose.TOOL_PLANNING,
            tools=(
                ToolDescriptor(
                    tool_id="tool-1",
                    name="write_file",
                    kind=ToolKind.NATIVE,
                    description="Writes files",
                    argument_schema={"type": "object"},
                ),
            ),
        )
    )

    assert result.tool_call == planned
    assert result.json_value == planned
    envelope = json.loads(captured["input"])
    assert envelope["purpose"] == "tool_planning"
    assert envelope["tools"][0]["tool_id"] == "tool-1"
    assert envelope["tools"][0]["tool_name"] == "write_file"


def test_codex_provider_missing_cli_raises_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: None)

    with pytest.raises(ValueError, match="openai_codex_cli_missing"):
        OpenAICodexProvider().generate_markdown(
            InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA)
        )


def test_codex_provider_auth_failure_raises_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="please login first")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_auth_required"):
        OpenAICodexProvider().generate_markdown(
            InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA)
        )


def test_codex_provider_reports_when_cli_upgrade_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="This model requires a newer version of Codex")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_cli_upgrade_required"):
        OpenAICodexProvider().generate_markdown(
            InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA)
        )


def test_codex_provider_timeout_raises_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=1)

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_timeout"):
        OpenAICodexProvider(OpenAICodexProviderConfig(timeout_seconds=1)).generate_markdown(
            InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA)
        )


def test_codex_provider_empty_stdout_raises_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_empty_response"):
        OpenAICodexProvider().generate_markdown(
            InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA)
        )


def test_codex_provider_invalid_json_raises_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="not json", stderr="")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_invalid_json"):
        OpenAICodexProvider().generate_json(
            InferenceRequest(prompt="Prompt", purpose=InferencePurpose.CRITERIA_REVIEW)
        )


def test_codex_provider_nonzero_exit_raises_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_exec_failed: exit_code=2"):
        OpenAICodexProvider().generate_markdown(
            InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA)
        )
