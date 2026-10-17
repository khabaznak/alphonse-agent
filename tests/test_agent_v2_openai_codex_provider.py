from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRequest
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import OpenAICodexProvider
from alphonse.agent_v2.core.inference import OpenAICodexProviderConfig
from alphonse.agent_v2.core.inference.openai_codex import _run_interruptible


def test_codex_provider_markdown_maps_stdout_to_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        captured["cwd"] = kwargs["cwd"]
        captured["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout="1.- [ ] File exists\n", stderr="")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    provider = OpenAICodexProvider(OpenAICodexProviderConfig())
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
    assert captured.get("timeout") is None
    envelope = json.loads(str(captured["input"]))
    assert envelope["purpose"] == "acceptance_criteria"
    assert envelope["project_id"] == "alpha"
    assert envelope["user"] == "alex"
    assert envelope["task_id"] == "task-1"
    assert envelope["prompt"] == "Generate acceptance criteria"


def test_codex_provider_ignores_environment_model(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_codex_provider_can_require_saved_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_CODEX_MODEL", "environment-model")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    provider = OpenAICodexProvider(OpenAICodexProviderConfig(require_explicit_model=True))
    with pytest.raises(ValueError, match="openai_codex_model_not_configured"):
        provider.generate_markdown(InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA))


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


@pytest.mark.parametrize("stderr", [
    "Review the user's authorization requirements.\nERROR: connection closed",
    "MCP startup failed: authentication failed\nERROR: connection closed",
    "For login help see the documentation.\nERROR: connection closed",
])
def test_codex_provider_does_not_mislabel_auth_mentions(monkeypatch, stderr):
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr=stderr))
    with pytest.raises(ValueError, match="openai_codex_exec_failed: exit_code=2"):
        OpenAICodexProvider().generate_markdown(InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA))


def test_codex_provider_reports_when_cli_upgrade_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="This model requires a newer version of Codex")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_cli_upgrade_required"):
        OpenAICodexProvider().generate_markdown(
            InferenceRequest(prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA)
        )


def test_codex_provider_reports_ambiguous_model_access_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ERROR: The model `gpt-5.5` does not exist or you do not have access to it.",
        )

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match=r"openai_codex_model_access_rejected: gpt-5\.5"):
        OpenAICodexProvider().generate_markdown(
            InferenceRequest(
                prompt="Prompt",
                purpose=InferencePurpose.ACCEPTANCE_CRITERIA,
                model_profile=ModelProfile(provider="openai_codex", model="gpt-5.5", profile_id="saved"),
            )
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


def test_codex_provider_passes_live_cancellation_to_interruptible_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = lambda: True
    captured = {}

    def fake_interruptible(command, **kwargs):
        captured["checker"] = kwargs["cancel_checker"]
        raise ValueError("inference_cancelled")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex._run_interruptible", fake_interruptible)

    with pytest.raises(ValueError, match="inference_cancelled"):
        OpenAICodexProvider().generate_markdown(InferenceRequest(
            prompt="Prompt", purpose=InferencePurpose.ACCEPTANCE_CRITERIA, cancel_checker=checker,
        ))

    assert captured["checker"] is checker


def test_interruptible_runner_delivers_large_prompt_after_initial_poll_timeout(tmp_path) -> None:
    prompt = "large prompt\n" * 100_000

    completed = _run_interruptible(
        [
            sys.executable,
            "-c",
            "import sys, time; time.sleep(0.2); data = sys.stdin.read(); print(len(data))",
        ],
        prompt=prompt,
        timeout_seconds=2,
        cwd=str(tmp_path),
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == str(len(prompt))
