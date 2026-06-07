from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

from alphonse.agent.cognition.providers.contracts import CanonicalCompleteWithToolsResult
from alphonse.agent.cognition.providers.contracts import require_canonical_single_tool_call_result


class OpenAICodexClient:
    def __init__(
        self,
        *,
        cli_bin: str = "codex",
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.cli_bin = cli_bin
        self.model = model
        self.timeout = timeout
        self.supports_tool_calls = True
        self.tool_result_message_style = "openai"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        prompt = (
            f"{system_prompt.strip()}\n\n"
            f"{user_prompt.strip()}"
        ).strip()
        output = self._run_codex(prompt)
        if not output:
            raise ValueError("openai_codex_empty_response")
        return output

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> CanonicalCompleteWithToolsResult:
        envelope = {
            "tool_choice": str(tool_choice or "auto").strip() or "auto",
            "tools": list(tools) if isinstance(tools, list) else [],
            "messages": list(messages) if isinstance(messages, list) else [],
        }
        prompt = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        output = self._run_codex(prompt)
        parsed = _try_parse_json_object(output)
        if not isinstance(parsed, dict):
            raise ValueError("openai_codex_invalid_tool_json")
        return require_canonical_single_tool_call_result(
            parsed,
            error_prefix="openai_codex_complete_with_tools_non_canonical",
        )

    def _run_codex(self, prompt: str) -> str:
        if not shutil.which(self.cli_bin):
            raise ValueError("openai_codex_cli_missing")
        command = [self.cli_bin, "exec", "--skip-git-repo-check"]
        if self.model:
            command.extend(["--model", self.model])
        try:
            with tempfile.TemporaryDirectory(prefix="alphonse-codex-") as workdir:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=workdir,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("openai_codex_timeout") from exc

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            text = f"{stdout}\n{stderr}".lower()
            if any(token in text for token in ("login", "auth", "authenticate", "unauthorized")):
                raise ValueError("openai_codex_auth_required")
            raise ValueError(f"openai_codex_exec_failed: exit_code={completed.returncode}")
        return stdout


def build_openai_codex_client_from_env() -> OpenAICodexClient:
    timeout = _parse_float(os.getenv("OPENAI_CODEX_TIMEOUT_SECONDS"), default=120.0)
    model = os.getenv("OPENAI_CODEX_MODEL") or None
    cli_bin = os.getenv("OPENAI_CODEX_CLI_BIN", "codex")
    return OpenAICodexClient(cli_bin=cli_bin, model=model, timeout=timeout)


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if not value:
        return None

    def _decode(candidate: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    if value.startswith("```"):
        trimmed = value.strip("`").strip()
        if trimmed.lower().startswith("json"):
            trimmed = trimmed[4:].strip()
        parsed = _decode(trimmed)
        if isinstance(parsed, dict):
            return parsed

    parsed = _decode(value)
    if isinstance(parsed, dict):
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
