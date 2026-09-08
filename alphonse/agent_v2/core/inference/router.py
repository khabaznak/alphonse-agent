"""Inference provider protocol and routing for v2."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

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
    ) -> None:
        self.provider = provider
        self.default_profile = default_profile

    def select_profile(self, request: InferenceRequest) -> ModelProfile:
        """Return the one model selected for the complete CAPD cycle."""
        _ = request
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
        # Callers cannot override the Desktop/TUI selection per project, node,
        # purpose, or request.
        return replace(request, model_profile=self.select_profile(request))

    @staticmethod
    def _with_result_profile(result: InferenceResult, profile: ModelProfile | None) -> InferenceResult:
        if result.model_profile is not None:
            return result
        return replace(result, model_profile=profile)
