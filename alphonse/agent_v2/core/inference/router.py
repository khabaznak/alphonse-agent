"""Inference provider protocol and routing for v2."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from alphonse.agent_v2.core.inference.models import InferencePurpose
from alphonse.agent_v2.core.inference.models import InferenceRequest
from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.inference.models import ModelProfile


class InferenceProvider(Protocol):
    """Provider boundary for model-backed CAPD node calls."""

    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        """Generate markdown or plain text content."""

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        """Generate JSON-compatible structured content."""

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        """Generate one planned tool call."""


class InferenceRouter:
    """Selects a model profile and delegates to an inference provider."""

    def __init__(
        self,
        *,
        provider: InferenceProvider,
        default_profile: ModelProfile,
        project_profiles: dict[str, ModelProfile] | None = None,
        purpose_profiles: dict[InferencePurpose, ModelProfile] | None = None,
    ) -> None:
        self.provider = provider
        self.default_profile = default_profile
        self.project_profiles = dict(project_profiles or {})
        self.purpose_profiles = dict(purpose_profiles or {})

    def select_profile(self, request: InferenceRequest) -> ModelProfile:
        if request.project_id and request.project_id in self.project_profiles:
            return self.project_profiles[request.project_id]
        if request.purpose in self.purpose_profiles:
            return self.purpose_profiles[request.purpose]
        return self.default_profile

    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        resolved = self._with_profile(request)
        return self._with_result_profile(self.provider.generate_markdown(resolved), resolved.model_profile)

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        resolved = self._with_profile(request)
        return self._with_result_profile(self.provider.generate_json(resolved), resolved.model_profile)

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        resolved = self._with_profile(request)
        return self._with_result_profile(self.provider.plan_tool_call(resolved), resolved.model_profile)

    def _with_profile(self, request: InferenceRequest) -> InferenceRequest:
        if request.model_profile is not None:
            return request
        return replace(request, model_profile=self.select_profile(request))

    @staticmethod
    def _with_result_profile(result: InferenceResult, profile: ModelProfile | None) -> InferenceResult:
        if result.model_profile is not None:
            return result
        return replace(result, model_profile=profile)
