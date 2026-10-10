from __future__ import annotations

from typing import Any

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest, InferenceResult
from alphonse.agent_v2.core.inference import InferenceRouter, ModelProfile
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.runtime import build_runtime_host


class _Provider:
    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(content="done", usage={"input_tokens": 12, "output_tokens": 2})

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(json_value={"done": True})

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        raise RuntimeError("provider unavailable")


def _request() -> InferenceRequest:
    tool = ToolDescriptor(
        tool_id="native.read",
        name="read",
        kind=ToolKind.NATIVE,
        argument_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    return InferenceRequest(
        prompt="Read one record",
        purpose=InferencePurpose.TOOL_PLANNING,
        task_id="task-1",
        project_id="home",
        tools=(tool,),
        metadata={"phase_id": "locate", "api_token": "do-not-log"},
    )


def test_inference_router_emits_bounded_success_and_failure_telemetry() -> None:
    events: list[dict[str, Any]] = []
    router = InferenceRouter(
        provider=_Provider(),
        default_profile=ModelProfile(provider="test", model="model", profile_id="profile"),
        telemetry_sink=events.append,
    )

    markdown_request = _request()
    router.generate_markdown(markdown_request)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        router.plan_tool_call(markdown_request)

    assert [event["status"] for event in events] == ["success", "failed"]
    success = events[0]
    assert success["purpose"] == "tool_planning"
    assert success["task_id"] == "task-1"
    assert success["tool_count"] == 1
    assert success["tool_schema_tokens_estimate"] > 0
    assert success["provider_usage"] == {"input_tokens": 12, "output_tokens": 2}
    assert success["metadata"]["api_token"] == "[redacted]"
    assert "Read one record" not in str(success)


def test_tool_invocation_emits_result_scope_and_failures() -> None:
    events: list[dict[str, Any]] = []
    registry = InMemoryToolRegistry()
    registry.register(ToolDefinition(
        descriptor=ToolDescriptor(
            tool_id="native.edit", name="edit", kind=ToolKind.NATIVE,
            argument_schema={"type": "object", "required": ["path"]},
        ),
        callable=lambda arguments: {"affected_paths": [arguments["path"]]},
    ))
    context = CoreLoopContext(
        messages=InMemoryMessageQueue(), tools=registry, telemetry_sink=events.append,
    )
    service = ToolInvocationService(context=context, task=TaskState(task_id="task", project_id="home"))

    service.execute_or_raise("native.edit", {"path": "backlog.md"})
    with pytest.raises(ValueError, match="tool_arguments_invalid"):
        service.execute_or_raise("native.edit", {})

    assert [event["status"] for event in events] == ["success", "failed"]
    assert events[0]["metadata"]["affected_paths"] == ["backlog.md"]
    assert events[1]["tool_id"] == "native.edit"


def test_runtime_collects_router_telemetry_without_discarding_injected_sink() -> None:
    injected: list[dict[str, Any]] = []
    router = InferenceRouter(
        provider=_Provider(),
        default_profile=ModelProfile(provider="test", model="model", profile_id="profile"),
        telemetry_sink=injected.append,
    )
    runtime = build_runtime_host(inference=router)

    runtime.core.inference.generate_markdown(_request())

    assert len(injected) == 1
    assert len(runtime.telemetry_events) == 1
    assert runtime.telemetry_events[0]["event_type"] == "inference"
