"""Deterministic inference provider for tests and early v2 wiring."""

from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.inference.models import InferencePurpose
from alphonse.agent_v2.core.inference.models import InferenceRequest
from alphonse.agent_v2.core.inference.models import InferenceResult


class StubInferenceProvider:
    """Configurable no-network provider used while real providers are absent."""

    def __init__(
        self,
        *,
        markdown_by_purpose: dict[InferencePurpose, str | None] | None = None,
        json_by_purpose: dict[InferencePurpose, Any] | None = None,
        tool_call: dict[str, Any] | None = None,
    ) -> None:
        self.markdown_by_purpose = dict(markdown_by_purpose or {})
        self.json_by_purpose = dict(json_by_purpose or {})
        self.tool_call = dict(tool_call) if isinstance(tool_call, dict) else None
        self.requests: list[InferenceRequest] = []

    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        content = self.markdown_by_purpose.get(request.purpose)
        return InferenceResult(content=str(content or ""), model_profile=request.model_profile)

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        return InferenceResult(json_value=self.json_by_purpose.get(request.purpose), model_profile=request.model_profile)

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        return InferenceResult(
            json_value=dict(self.tool_call) if self.tool_call is not None else None,
            tool_call=dict(self.tool_call) if self.tool_call is not None else None,
            model_profile=request.model_profile,
        )
