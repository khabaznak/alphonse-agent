"""Inference provider protocol and routing for v2."""

from __future__ import annotations

import json
from dataclasses import replace
from time import monotonic
from typing import Any, Callable, Protocol

from alphonse.agent_v2.core.inference.models import InferenceRequest
from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.inference.models import ModelProfile
from alphonse.agent_v2.core.telemetry import InferenceTelemetryEvent, approximate_tokens, json_chars, safe_telemetry_value


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
        telemetry_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.provider = provider
        self.default_profile = default_profile
        self.telemetry_sink = telemetry_sink

    def select_profile(self, request: InferenceRequest) -> ModelProfile:
        """Return the one model selected for the complete CAPD cycle."""
        _ = request
        return self.default_profile

    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        return self._invoke("generate_markdown", request)

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        return self._invoke("generate_json", request)

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        return self._invoke("plan_tool_call", request)

    def _invoke(self, method: str, request: InferenceRequest) -> InferenceResult:
        resolved = self._with_profile(request)
        started = monotonic()
        try:
            result = getattr(self.provider, method)(resolved)
        except Exception as exc:
            self._emit(resolved, None, started, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        normalized = self._with_result_profile(result, resolved.model_profile)
        self._emit(resolved, normalized, started, status="success", error="")
        return normalized

    def _emit(
        self,
        request: InferenceRequest,
        result: InferenceResult | None,
        started: float,
        *,
        status: str,
        error: str,
    ) -> None:
        if self.telemetry_sink is None:
            return
        schemas = [tool.argument_schema for tool in request.tools]
        schema_chars = json_chars(schemas) if schemas else 0
        output = ""
        if result is not None:
            output = result.content or (
                json.dumps(result.json_value, ensure_ascii=False, default=str) if result.json_value is not None else ""
            )
        profile = request.model_profile
        event = InferenceTelemetryEvent(
            purpose=request.purpose.value,
            task_id=str(request.task_id or ""),
            project_id=request.project_id,
            provider=str(profile.provider if profile else ""),
            model=str(profile.model if profile else ""),
            profile_id=str(profile.profile_id if profile else ""),
            input_chars=len(request.prompt),
            input_tokens_estimate=approximate_tokens(request.prompt),
            output_chars=len(output),
            output_tokens_estimate=approximate_tokens(output),
            tool_count=len(request.tools),
            tool_schema_chars=schema_chars,
            tool_schema_tokens_estimate=0 if schema_chars == 0 else max(1, (schema_chars + 3) // 4),
            duration_ms=max(0, round((monotonic() - started) * 1000)),
            status=status,
            error=str(safe_telemetry_value(error)),
            provider_usage=safe_telemetry_value(dict(result.usage)) if result is not None else {},
            metadata=safe_telemetry_value(dict(request.metadata)),
        )
        self.telemetry_sink(event.to_dict())

    def _with_profile(self, request: InferenceRequest) -> InferenceRequest:
        # Callers cannot override the Desktop/TUI selection per project, node,
        # purpose, or request.
        return replace(request, model_profile=self.select_profile(request))

    @staticmethod
    def _with_result_profile(result: InferenceResult, profile: ModelProfile | None) -> InferenceResult:
        if result.model_profile is not None:
            return result
        return replace(result, model_profile=profile)
