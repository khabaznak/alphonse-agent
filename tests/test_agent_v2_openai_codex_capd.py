from __future__ import annotations

import json
import pytest
from types import SimpleNamespace

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import OpenAICodexProvider
from alphonse.agent_v2.core.inference import OpenAICodexProviderConfig
from alphonse.agent_v2.evaluation.legacy_pdca.nodes.check_node import check_node
from alphonse.agent_v2.evaluation.legacy_pdca.nodes.plan_node import plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_plan_node_uses_codex_provider_for_acceptance_criteria(monkeypatch) -> None:
    captured = []
    router = _router()

    def fake_run(command, **kwargs):
        envelope = json.loads(kwargs["input"])
        captured.append(envelope)
        output = "1.- [ ] File exists" if envelope["purpose"] == "acceptance_criteria" else json.dumps({"tool_id": "tool-1", "arguments": {}})
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    task = TaskState(goal="Write the file", user="alex", check_verdict="new")
    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), inference=router, tools=_ToolRegistry()))

    assert task.acceptance_criteria_md == "1.- [ ] File exists"
    assert task.metadata["acceptance_criteria_model_profile"] == "plus"
    envelope = captured[0]
    assert envelope["purpose"] == "acceptance_criteria"
    assert "tools" not in envelope


def test_check_node_uses_system_one_without_codex_review(monkeypatch, pdca_decisions) -> None:
    router = _router()
    pdca_decisions.satisfy = True

    def unexpected_provider_call(*args, **kwargs):
        raise AssertionError("Bounded evidence review belongs to System One")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", unexpected_provider_call)

    task = TaskState(
        goal="Write the file",
        user="alex",
        acceptance_criteria_md="1.- [ ] File exists",
    )
    task.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {},
            "internal_state": "Writing.",
            "execution": {"status": "success", "output": {"ok": True}},
        }
    )

    check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), inference=router, system_one=pdca_decisions))

    assert task.acceptance_criteria_md == "1.- [x] File exists"
    assert task.metadata["system_one_check_review"]["updates"][0]["evidence_refs"] == ["tool-call:plan-call-1"]
    assert pdca_decisions.calls[0][0] == "check"


def test_plan_node_uses_codex_provider_with_exposed_tools(monkeypatch) -> None:
    captured: dict[str, str] = {}
    router = _router()
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

    task = TaskState(goal="Write the file", user="alex", acceptance_criteria_md="1.- [ ] File exists")
    plan_node(
        task,
        context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry(), inference=router),
    )

    assert task.metadata["tool_call_planning_llm_stubbed"] is False
    assert json.loads(task.plan_json)[0]["tool_id"] == "tool-1"
    envelope = json.loads(captured["input"])
    assert envelope["purpose"] == "tool_planning"
    assert envelope["tools"][0]["tool_id"] == "tool-1"
    assert envelope["tools"][0]["read_only"] is False


def _router() -> InferenceRouter:
    return InferenceRouter(
        provider=OpenAICodexProvider(OpenAICodexProviderConfig()),
        default_profile=ModelProfile(provider="openai_codex", model="gpt-plus", profile_id="plus"),
    )


class _ToolRegistry:
    def list(self) -> tuple[ToolDescriptor, ...]:
        return (
            ToolDescriptor(
                tool_id="tool-1",
                name="write_file",
                kind=ToolKind.NATIVE,
                description="Writes a file",
            ),
        )

    def execute(self, tool_id: str, arguments: dict[str, object]) -> dict[str, bool]:
        return {"ok": True}
