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
from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import act_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.check_node import check_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_act_node_uses_codex_provider_for_acceptance_criteria(monkeypatch) -> None:
    captured: dict[str, str] = {}
    router = _router()

    def fake_run(command, **kwargs):
        captured["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout="1.- [ ] File exists", stderr="")

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

    task = TaskState(goal="Write the file", user="alex", check_verdict="new")
    act_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), inference=router))

    # Plan now establishes acceptance criteria; Act only routes a new task.
    assert task.acceptance_criteria_md == '- (none)'
    assert task.metadata['act_route'] == 'plan'
    assert captured == {}
    from alphonse.agent_v2.core.intelligence.acceptance_planning import plan_acceptance_contract
    assert plan_acceptance_contract(task, CoreLoopContext(messages=InMemoryMessageQueue(), inference=router))
    assert task.acceptance_criteria_md == '1.- [ ] File exists'
    envelope = json.loads(captured['input'])
    assert envelope['purpose'] == 'acceptance_criteria'
    assert 'tools' not in envelope


def test_check_node_uses_codex_provider_for_criteria_review(monkeypatch) -> None:
    captured: dict[str, str] = {}
    router = _router()

    def fake_run(command, **kwargs):
        captured["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "updates": [
                        {
                            "criterion_id": "ac-1",
                            "status": "satisfied",
                            "evidence_refs": ["tool-call:plan-call-1"],
                        }
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent_v2.core.inference.openai_codex.subprocess.run", fake_run)

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

    from alphonse.agent_v2.system_one import SystemOneUnavailableError
    # Bounded evidence classification belongs to Jev; missing Jev fails closed.
    with pytest.raises(SystemOneUnavailableError, match='check_criteria_review_unavailable'):
        check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), inference=router))
    assert task.acceptance_criteria_md == '1.- [ ] File exists'
    assert captured == {}


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
