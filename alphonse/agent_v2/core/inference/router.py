"""Inference provider protocol and routing for v2."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from time import monotonic
from typing import Any, Callable, Protocol

from alphonse.agent_v2.core.inference.models import InferenceRequest
from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.inference.models import ModelProfile
from alphonse.agent_v2.core.telemetry import InferenceTelemetryEvent, approximate_tokens, json_chars, safe_telemetry_value


logger = logging.getLogger(__name__)

# Only exact, product-defined errors can produce a more specific diagnostic.
# Provider messages and even exception class names may contain household data.
_CODEX_FAILURES = {
    'openai_codex_model_not_configured': 'model_not_configured',
    'openai_codex_cli_missing': 'cli_missing',
    'openai_codex_subscription_auth_unverified': 'subscription_auth_unverified',
    'openai_codex_auth_required': 'authentication_required',
    'openai_codex_cli_upgrade_required': 'cli_upgrade_required',
    'openai_codex_timeout': 'provider_timeout',
    'openai_codex_empty_response': 'empty_response',
    'openai_codex_invalid_json': 'invalid_response',
    'inference_cancelled': 'cancelled',
}


def _simple_loop_failure_category(exc: Exception) -> str:
    if type(exc) is ValueError and len(exc.args) == 1 and type(exc.args[0]) is str:
        return _CODEX_FAILURES.get(exc.args[0], 'provider_error')
    return 'provider_error'


def _simple_loop_failure_detail(exc: Exception) -> str | None:
    """Report only a fixed parse stage, never provider text or arbitrary attrs."""
    if _simple_loop_failure_category(exc) != 'invalid_response':
        return None
    stage = getattr(exc, 'response_stage', None)
    allowed = {'outer_object', 'arguments_not_string', 'arguments_invalid_json', 'arguments_not_object'}
    return stage if type(stage) is str and stage in allowed else None


class InferenceProvider(Protocol):
    """Provider boundary for model-backed CAPD node calls."""

    supports_image_input: bool

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
        from alphonse.agent_v2.core.context_manifests import begin_decision, finish_decision
        resolved = self._with_profile(request)
        sections = {"prompt": resolved.prompt}
        if resolved.tools:
            sections["tool_descriptors"] = [{"name": t.name, "description": t.description, "argument_schema": t.argument_schema} for t in resolved.tools]
        if resolved.output_schema is not None:
            sections["output_schema"] = resolved.output_schema
        manifest = begin_decision(
            purpose=resolved.purpose.value,
            sections=sections,
            tools=resolved.tools, images=resolved.images, projections=resolved.context_manifests,
            sources=(*resolved.input_sources, *({'source_id': image.asset_id, 'source_type': 'image_input', 'reason': 'attached'} for image in resolved.images)),
        )
        if manifest is not None:
            resolved = replace(resolved, metadata={**resolved.metadata, "decision_id": manifest["decision_id"]})
        started = monotonic()
        try:
            if resolved.images and not getattr(self.provider, "supports_image_input", False):
                raise ValueError("inference_provider_image_input_unsupported")
            result = getattr(self.provider, method)(resolved)
        except Exception as exc:
            finish_decision(manifest, failed=True)
            # Provider exception strings can contain the raw model response.
            if resolved.purpose.value == "tactical_action":
                error = "tactical_provider_error"
            elif resolved.purpose.value == "simple_loop_action":
                # A provider exception may contain model output or source data.
                # The simple-loop chooser sanitizes what it returns, but router
                # telemetry is emitted before that outer handler runs.
                category = _simple_loop_failure_category(exc)
                error = f"simple_loop_{category}"
                # No exception text, traceback, prompts, responses or credentials.
                logger.warning("simple_loop_inference_failed category=%s", category)
                detail = _simple_loop_failure_detail(exc)
                if detail is not None:
                    logger.warning("simple_loop_inference_invalid_response stage=%s", detail)
            else:
                error = f"{type(exc).__name__}: {exc}"
            self._emit(resolved, None, started, status="failed", error=error)
            raise
        normalized = self._with_result_profile(result, resolved.model_profile)
        finish_decision(manifest)
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
            provider_usage=(safe_telemetry_value(dict(result.usage)) if result is not None
                            and request.purpose.value != "tactical_action" else {}),
            metadata=safe_telemetry_value(dict(request.metadata)),
        )
        self.telemetry_sink(event.to_dict())

    def _with_profile(self, request: InferenceRequest) -> InferenceRequest:
        # Callers cannot override the Desktop selection per project, node,
        # purpose, or request.
        return replace(request, model_profile=self.select_profile(request))

    @staticmethod
    def _with_result_profile(result: InferenceResult, profile: ModelProfile | None) -> InferenceResult:
        if result.model_profile is not None:
            return result
        return replace(result, model_profile=profile)
