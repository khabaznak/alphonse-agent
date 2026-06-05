from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from alphonse.agent.cognition.providers.contracts import CanonicalCompleteWithToolsResult
from alphonse.agent.cognition.providers.contracts import require_canonical_single_tool_call_result


class GitHubCopilotClient:
    def __init__(
        self,
        *,
        github_token_env: str = "COPILOT_GITHUB_TOKEN",
        client_id_env: str = "GITHUB_COPILOT_CLIENT_ID",
        model: str | None = None,
        timeout: float = 120.0,
        node_bin: str = "node",
        bridge_path: Path | None = None,
    ) -> None:
        self.github_token_env = github_token_env
        self.client_id_env = client_id_env
        self.model = model
        self.timeout = timeout
        self.node_bin = node_bin
        self.bridge_path = bridge_path or Path(__file__).with_name("copilot_bridge.mjs")
        self.supports_tool_calls = True
        self.tool_result_message_style = "openai"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        body = self._run_bridge(
            {
                "mode": "complete",
                "systemPrompt": system_prompt,
                "userPrompt": user_prompt,
            }
        )
        content = body.get("content") if isinstance(body, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("github_copilot_empty_response")
        return content.strip()

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> CanonicalCompleteWithToolsResult:
        body = self._run_bridge(
            {
                "mode": "complete_with_tools",
                "messages": messages,
                "tools": tools,
                "toolChoice": tool_choice,
            }
        )
        return require_canonical_single_tool_call_result(
            body,
            error_prefix="github_copilot_complete_with_tools_non_canonical",
        )

    def _run_bridge(self, payload: dict[str, Any]) -> dict[str, Any]:
        token = os.getenv(self.github_token_env)
        if not token:
            raise ValueError("github_copilot_token_missing")
        if not shutil.which(self.node_bin):
            raise ValueError("github_copilot_node_missing")
        request = dict(payload)
        request["githubToken"] = token
        request["clientId"] = os.getenv(self.client_id_env) or ""
        request["model"] = self.model
        try:
            completed = subprocess.run(
                [self.node_bin, str(self.bridge_path)],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("github_copilot_timeout") from exc

        if completed.returncode != 0:
            code = _extract_bridge_error_code(completed.stderr) or "github_copilot_bridge_failed"
            raise ValueError(code)
        try:
            parsed = json.loads(completed.stdout or "{}")
        except ValueError as exc:
            raise ValueError("github_copilot_invalid_bridge_json") from exc
        if not isinstance(parsed, dict):
            raise ValueError("github_copilot_invalid_bridge_json")
        if isinstance(parsed.get("error"), dict):
            code = str(parsed["error"].get("code") or "github_copilot_bridge_failed")
            raise ValueError(code)
        return parsed


def build_github_copilot_client_from_env() -> GitHubCopilotClient:
    timeout = _parse_float(os.getenv("COPILOT_TIMEOUT_SECONDS"), default=120.0)
    return GitHubCopilotClient(
        github_token_env=os.getenv("COPILOT_GITHUB_TOKEN_ENV", "COPILOT_GITHUB_TOKEN"),
        client_id_env=os.getenv("GITHUB_COPILOT_CLIENT_ID_ENV", "GITHUB_COPILOT_CLIENT_ID"),
        model=os.getenv("COPILOT_MODEL") or None,
        timeout=timeout,
        node_bin=os.getenv("GITHUB_COPILOT_NODE_BIN", "node"),
    )


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _extract_bridge_error_code(stderr: str) -> str | None:
    text = str(stderr or "").strip()
    if not text:
        return None
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        error = parsed.get("error") if isinstance(parsed, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str) and code.strip():
            return code.strip()
    lowered = text.lower()
    if "unauthorized" in lowered or "forbidden" in lowered or "auth" in lowered:
        return "github_copilot_auth_failed"
    return None
