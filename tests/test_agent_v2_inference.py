from __future__ import annotations

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRequest
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider


def test_inference_router_uses_one_profile_across_projects_and_purposes() -> None:
    default = ModelProfile(provider="openai", model="gpt", profile_id="default")
    provider = StubInferenceProvider(markdown_by_purpose={InferencePurpose.CRITERIA_REVIEW: "done"})
    router = InferenceRouter(provider=provider, default_profile=default)

    result = router.generate_markdown(
        InferenceRequest(prompt="Review", purpose=InferencePurpose.CRITERIA_REVIEW, project_id="stoic")
    )

    assert result.content == "done"
    assert result.model_profile == default
    assert provider.requests[0].model_profile == default


def test_inference_router_rejects_request_level_model_override() -> None:
    selected = ModelProfile(provider="openai", model="selected", profile_id="saved")
    override = ModelProfile(provider="openai", model="other", profile_id="request")
    provider = StubInferenceProvider(markdown_by_purpose={InferencePurpose.ACCEPTANCE_CRITERIA: "done"})
    router = InferenceRouter(provider=provider, default_profile=selected)
    result = router.generate_markdown(InferenceRequest(prompt="Act", purpose=InferencePurpose.ACCEPTANCE_CRITERIA, model_profile=override))
    assert result.model_profile == selected
    assert provider.requests[0].model_profile == selected


def test_inference_router_falls_back_to_default_profile() -> None:
    default = ModelProfile(provider="openai", model="gpt", profile_id="default")
    provider = StubInferenceProvider(markdown_by_purpose={InferencePurpose.ACCEPTANCE_CRITERIA: "criteria"})
    router = InferenceRouter(provider=provider, default_profile=default)

    result = router.generate_markdown(
        InferenceRequest(prompt="Act", purpose=InferencePurpose.ACCEPTANCE_CRITERIA, project_id="unknown")
    )

    assert result.content == "criteria"
    assert result.model_profile == default


def test_stub_provider_returns_tool_call_for_plan() -> None:
    profile = ModelProfile(provider="openai", model="gpt", profile_id="default")
    tool_call = {
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {"path": "a.txt"},
        "internal_state": "Writing a file.",
    }
    router = InferenceRouter(
        provider=StubInferenceProvider(tool_call=tool_call),
        default_profile=profile,
    )

    result = router.plan_tool_call(InferenceRequest(prompt="Plan", purpose=InferencePurpose.TOOL_PLANNING))

    assert result.tool_call == tool_call
    assert result.json_value == tool_call
    assert result.model_profile == profile
