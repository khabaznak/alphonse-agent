"""ChatGPT Plus/Codex CLI inference provider for Alphonse v2."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.inference.models import InferenceRequest
from alphonse.agent_v2.core.inference.models import InferenceResult


@dataclass(frozen=True)
class OpenAICodexProviderConfig:
    """Configuration for the Codex CLI subscription provider."""

    cli_bin: str = "codex"
    timeout_seconds: float = 120.0
    ephemeral: bool = False
    require_explicit_model: bool = False


class OpenAICodexProvider:
    """Inference provider backed by the official Codex CLI subscription flow."""

    def __init__(self, config: OpenAICodexProviderConfig | None = None) -> None:
        self.config = config or OpenAICodexProviderConfig()

    def generate_markdown(self, request: InferenceRequest) -> InferenceResult:
        output = self._run_codex(_markdown_envelope(request), request)
        return InferenceResult(content=output, model_profile=request.model_profile, raw_response=output)

    def generate_json(self, request: InferenceRequest) -> InferenceResult:
        output = self._run_codex(_json_envelope(request), request)
        parsed = _try_parse_json_object(output)
        if parsed is None:
            raise ValueError("openai_codex_invalid_json")
        return InferenceResult(json_value=parsed, model_profile=request.model_profile, raw_response=output)

    def plan_tool_call(self, request: InferenceRequest) -> InferenceResult:
        output = self._run_codex(_tool_planning_envelope(request), request)
        parsed = _try_parse_json_object(output)
        if parsed is None:
            raise ValueError("openai_codex_invalid_tool_json")
        return InferenceResult(
            json_value=parsed,
            tool_call=parsed,
            model_profile=request.model_profile,
            raw_response=output,
        )

    def _run_codex(self, prompt: str, request: InferenceRequest) -> str:
        model = _model_for_request(request)
        if self.config.require_explicit_model and not model:
            raise ValueError("openai_codex_model_not_configured")
        cli_bin = self.config.cli_bin
        if not shutil.which(cli_bin):
            raise ValueError("openai_codex_cli_missing")

        command = [cli_bin, "exec", "--skip-git-repo-check"]
        if self.config.ephemeral:
            command.append("--ephemeral")
        if model:
            command.extend(["--model", model])

        try:
            with tempfile.TemporaryDirectory(prefix="alphonse-codex-") as workdir:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    cwd=workdir,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("openai_codex_timeout") from exc

        stdout = str(completed.stdout or "").strip()
        stderr = str(completed.stderr or "").strip()
        if completed.returncode != 0:
            text = f"{stdout}\n{stderr}".lower()
            if "requires a newer version of codex" in text:
                raise ValueError("openai_codex_cli_upgrade_required")
            # CLI output can include prompts and optional MCP startup warnings.
            # A mention of "auth" or "login" is not proof that model auth failed.
            auth_failures = (
                "please login first", "please log in", "not logged in",
                "401 unauthorized", "authentication failed", "refresh_token_reused",
                "refresh token has already been used", "invalid_api_key",
            )
            if any(
                marker in line
                for line in text.splitlines()
                if "mcp" not in line
                for marker in auth_failures
            ):
                raise ValueError("openai_codex_auth_required")
            raise ValueError(f"openai_codex_exec_failed: exit_code={completed.returncode}")
        if not stdout:
            raise ValueError("openai_codex_empty_response")
        return stdout


def _markdown_envelope(request: InferenceRequest) -> str:
    return _render_envelope(
        request,
        instructions=(
            "Return only the requested markdown/text result. "
            "Do not wrap the answer in code fences unless the prompt explicitly requires it."
        ),
    )


def _json_envelope(request: InferenceRequest) -> str:
    return _render_envelope(
        request,
        instructions="Return one valid JSON object only. Do not include commentary or markdown fences.",
    )


def _tool_planning_envelope(request: InferenceRequest) -> str:
    return _render_envelope(
        request,
        instructions=(
            "Return one valid JSON object describing one bounded execution phase. "
            "Use one direct tool call or, when the prompt says program mode is available, a multi-tool program. "
            "For direct mode, the object must contain execution_mode='direct', tool_id, tool_name, arguments, and internal_state. "
            "For program mode, the object must contain execution_mode='program', a program object with language='python' and source, and internal_state."
        ),
        tools=request.tools,
    )


def _render_envelope(
    request: InferenceRequest,
    *,
    instructions: str,
    tools: tuple[ToolDescriptor, ...] = (),
) -> str:
    envelope: dict[str, Any] = {
        "provider_contract": "alphonse_agent_v2_inference",
        "purpose": request.purpose.value,
        "project_id": request.project_id,
        "user": request.user or "",
        "task_id": request.task_id or "",
        "instructions": instructions,
        "prompt": request.prompt,
    }
    if tools:
        envelope["tools"] = [_tool_descriptor_to_dict(tool) for tool in tools]
    if request.metadata:
        envelope["metadata"] = dict(request.metadata)
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2)


def _tool_descriptor_to_dict(tool: ToolDescriptor) -> dict[str, Any]:
    return {
        "tool_id": tool.tool_id,
        "tool_name": tool.name,
        "kind": tool.kind.value,
        "description": tool.description,
        "argument_schema": dict(tool.argument_schema),
        "capabilities": list(tool.capabilities),
        "tags": list(tool.tags),
        "read_only": tool.read_only,
    }


def _model_for_request(request: InferenceRequest) -> str | None:
    if request.model_profile is not None:
        # The router's saved profile is authoritative for every CAPD request.
        return request.model_profile.model.strip() or None
    return None


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if not value:
        return None

    def decode(candidate: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    if value.startswith("```"):
        trimmed = value.strip("`").strip()
        if trimmed.lower().startswith("json"):
            trimmed = trimmed[4:].strip()
        parsed = decode(trimmed)
        if parsed is not None:
            return parsed

    parsed = decode(value)
    if parsed is not None:
        return parsed

    decoder = json.JSONDecoder()
    idx = value.find("{")
    while idx >= 0:
        try:
            parsed_obj, _end = decoder.raw_decode(value[idx:])
        except ValueError:
            idx = value.find("{", idx + 1)
            continue
        if isinstance(parsed_obj, dict):
            return parsed_obj
        idx = value.find("{", idx + 1)
    return None
