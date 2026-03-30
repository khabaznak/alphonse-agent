from __future__ import annotations

import json
from typing import Any

import requests

from alphonse.agent.cognition.providers.contracts import CanonicalCompleteWithToolsResult
from alphonse.agent.cognition.providers.contracts import require_canonical_single_tool_call_result


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "mistral:7b-instruct",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.supports_tool_calls = True
        self.tool_result_message_style = "ollama"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()["message"]["content"]

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> CanonicalCompleteWithToolsResult:
        # Keep signature parity with other providers; Ollama /api/chat does not currently
        # consume OpenAI-style tool_choice reliably across models.
        _ = tool_choice
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        message = body.get("message") if isinstance(body, dict) else None
        content = ""
        raw_tool_calls: list[Any] = []
        if isinstance(message, dict):
            msg_content = message.get("content")
            if isinstance(msg_content, str):
                content = msg_content
            rtc = message.get("tool_calls")
            if isinstance(rtc, list):
                raw_tool_calls = rtc
        tool_call = _extract_first_canonical_tool_call(raw_tool_calls)
        out: dict[str, Any] = {}
        if content:
            out["content"] = content
            planner_intent = _extract_planner_intent_from_canonical_json_content(content)
            if isinstance(planner_intent, str) and planner_intent.strip():
                out["planner_intent"] = planner_intent.strip()[:160]
        if isinstance(tool_call, dict):
            out["tool_call"] = tool_call
        return _canonical_single_tool_call_result_or_raise(out)


def _extract_first_canonical_tool_call(raw_tool_calls: list[Any]) -> dict[str, Any] | None:
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        function_obj = item.get("function")
        if not isinstance(function_obj, dict):
            continue
        tool_name = str(function_obj.get("name") or "").strip()
        if not tool_name:
            continue
        raw_args = function_obj.get("arguments")
        args = _parse_tool_args(raw_args)
        return {
            "kind": "call_tool",
            "tool_name": tool_name,
            "args": args,
        }
    return None


def _parse_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extract_planner_intent_from_canonical_json_content(content: str) -> str | None:
    parsed = _try_parse_json_object(content)
    if not isinstance(parsed, dict):
        return None
    tool_call = parsed.get("tool_call")
    if not isinstance(tool_call, dict):
        return None
    if str(tool_call.get("kind") or "").strip() != "call_tool":
        return None
    tool_name = str(tool_call.get("tool_name") or "").strip()
    args = tool_call.get("args")
    if not tool_name or not isinstance(args, dict):
        return None
    planner_intent = parsed.get("planner_intent")
    if not isinstance(planner_intent, str):
        return None
    text = planner_intent.strip()
    return text[:160] if text else None


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if not value:
        return None
    decoder = json.JSONDecoder()

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

    first_brace = value.find("{")
    if first_brace >= 0:
        idx = first_brace
        while idx >= 0 and idx < len(value):
            try:
                parsed_obj, _end = decoder.raw_decode(value[idx:])
            except ValueError:
                idx = value.find("{", idx + 1)
                continue
            if isinstance(parsed_obj, dict):
                return parsed_obj
            idx = value.find("{", idx + 1)
    return None


def _canonical_single_tool_call_result_or_raise(raw: Any) -> CanonicalCompleteWithToolsResult:
    return require_canonical_single_tool_call_result(
        raw,
        error_prefix="ollama_complete_with_tools_non_canonical",
    )
