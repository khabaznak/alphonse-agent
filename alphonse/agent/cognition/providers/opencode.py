from __future__ import annotations

import json
import os
from typing import Any

import requests

from alphonse.agent.cognition.providers.contracts import CanonicalCompleteWithToolsResult
from alphonse.agent.cognition.providers.contracts import require_canonical_single_tool_call_result


class OpenCodeClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:4096",
        model: str = "openai/gpt-5.1-codex",
        timeout: float = 120.0,
        api_key_env: str = "OPENCODE_API_KEY",
        username_env: str = "OPENCODE_SERVER_USERNAME",
        password_env: str = "OPENCODE_SERVER_PASSWORD",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key_env = api_key_env
        self.username_env = username_env
        self.password_env = password_env
        self._session_id: str | None = None
        self.supports_tool_calls = True
        self.tool_result_message_style = "openai"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._complete_via_session_api(system_prompt, user_prompt)

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> CanonicalCompleteWithToolsResult:
        raw = self._complete_with_tools_via_session_api(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return _canonical_single_tool_call_result_or_raise(raw)

    def _complete_with_tools_via_session_api(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
    ) -> dict[str, Any]:
        headers, auth = _build_auth(self.api_key_env, self.username_env, self.password_env)
        session_id = self._ensure_session(headers=headers, auth=auth)
        tool_timeout_raw = os.getenv("OPENCODE_TOOL_CALL_TIMEOUT_SECONDS")
        tool_timeout = self.timeout
        if tool_timeout_raw:
            try:
                tool_timeout = max(5.0, float(tool_timeout_raw))
            except ValueError:
                tool_timeout = self.timeout

        rendered = _render_session_transport_payload_text(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        payload = {
            "parts": [{"type": "text", "text": rendered}],
        }
        model_payload = _model_for_session_payload(self.model)
        if model_payload:
            payload["model"] = model_payload
        response = requests.post(
            f"{self.base_url}/session/{session_id}/message",
            headers=headers,
            auth=auth,
            json=payload,
            timeout=tool_timeout,
        )
        _raise_for_status_with_body(
            response,
            "OpenCode session tool-call message failed",
        )
        body = _read_json_response(response)
        content, tool_call = _extract_session_tool_payload(body)
        planner_intent: str | None = None
        if content:
            canonical_tool_call, canonical_planner_intent, canonical_content = (
                _extract_canonical_from_text_content(content)
            )
            if isinstance(canonical_tool_call, dict):
                tool_call = canonical_tool_call
            if isinstance(canonical_planner_intent, str):
                planner_intent = canonical_planner_intent
            if canonical_content is not None:
                content = canonical_content
        out: dict[str, Any] = {}
        if content:
            out["content"] = content
        if isinstance(tool_call, dict):
            out["tool_call"] = tool_call
        if isinstance(planner_intent, str) and planner_intent.strip():
            out["planner_intent"] = planner_intent.strip()[:160]
        return out

    def _complete_via_session_api(self, system_prompt: str, user_prompt: str) -> str:
        headers, auth = _build_auth(self.api_key_env, self.username_env, self.password_env)
        session_id = self._ensure_session(headers=headers, auth=auth)
        payload = {
            "system": system_prompt,
            "parts": [{"type": "text", "text": user_prompt}],
        }
        model_payload = _model_for_session_payload(self.model)
        if model_payload:
            payload["model"] = model_payload
        response = requests.post(
            f"{self.base_url}/session/{session_id}/message",
            headers=headers,
            auth=auth,
            json=payload,
            timeout=self.timeout,
        )
        _raise_for_status_with_body(
            response,
            "OpenCode session message failed",
        )
        body = _read_json_response(response)
        content = _extract_session_message_content(body) or _extract_message_content(body)
        if not content:
            raise ValueError("OpenCode session response missing assistant content")
        return content

    def _ensure_session(self, *, headers: dict[str, str], auth: tuple[str, str] | None) -> str:
        if self._session_id:
            return self._session_id
        payload: dict[str, Any] = {}
        response = requests.post(
            f"{self.base_url}/session",
            headers=headers,
            auth=auth,
            json=payload,
            timeout=self.timeout,
        )
        _raise_for_status_with_body(response, "OpenCode session creation failed")
        body = _read_json_response(response)
        session_id = _extract_session_id(body)
        if not session_id:
            raise ValueError("OpenCode session creation did not return an id")
        self._session_id = session_id
        return session_id


def _build_auth(
    api_key_env: str,
    username_env: str,
    password_env: str,
) -> tuple[dict[str, str], tuple[str, str] | None]:
    headers = {"Content-Type": "application/json"}
    auth: tuple[str, str] | None = None
    api_key = os.getenv(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        username = os.getenv(username_env, "opencode")
        password = os.getenv(password_env)
        if password:
            auth = (username, password)
    return headers, auth


def _raise_for_status_with_body(response: requests.Response, prefix: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = (response.text or "").strip()
        snippet = body[:1000]
        raise ValueError(
            f"{prefix}. status={response.status_code} body={snippet!r}"
        ) from exc


def _read_json_response(response: requests.Response) -> dict[str, Any]:
    try:
        parsed = response.json() # TODO: when would it not parse and still is worth trying to extract strngs? 
    except ValueError:
        text = (response.text or "").strip()
        if text.startswith("data:"):
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    parsed = json.loads(chunk)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"OpenCode response was not JSON. status={response.status_code} body={text[:200]!r}"
                )
        else:
            raise ValueError(
                f"OpenCode response was not JSON. status={response.status_code} body={text[:200]!r}"
            )
    if not isinstance(parsed, dict):
        raise ValueError("OpenCode response root must be a JSON object")
    return parsed


def _model_for_session_payload(model: str) -> dict[str, str] | None:
    value = str(model or "").strip()
    if not value or "/" not in value:
        return None
    provider_id, model_id = value.split("/", 1)
    provider_id = provider_id.strip()
    model_id = model_id.strip()
    if not provider_id or not model_id:
        return None
    return {"providerID": provider_id, "modelID": model_id}


def _extract_session_id(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    for key in ("id", "ID", "session_id", "sessionID"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value
    info = body.get("info")
    if isinstance(info, dict):
        for key in ("id", "ID", "session_id", "sessionID"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _extract_session_message_content(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    parts = body.get("parts")
    if isinstance(parts, list):
        chunks: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
                continue
            content = part.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
        if chunks:
            return "\n".join(chunks)
    return None


def _extract_message_content(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    message = body.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    content = body.get("content")
    if isinstance(content, str) and content.strip():
        return content
    output = body.get("output_text")
    if isinstance(output, str) and output.strip():
        return output
    return None


def _render_session_transport_payload_text(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str,
) -> str:
    envelope = {
        "transport": {
            "mode": "session_tool_call",
            "version": 1,
        },
        "tool_choice": str(tool_choice or "auto").strip() or "auto",
        "tools": list(tools) if isinstance(tools, list) else [],
        "messages": list(messages) if isinstance(messages, list) else [],
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _extract_session_tool_payload(body: Any) -> tuple[str, dict[str, Any] | None]:
    content = ""
    tool_call: dict[str, Any] | None = None
    parts = body.get("parts") if isinstance(body, dict) else None
    if isinstance(parts, list):
        for idx, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip().lower()
            if part_type == "tool" and tool_call is None:
                tool_name = str(part.get("tool") or "").strip()
                if tool_name:
                    state = part.get("state") if isinstance(part.get("state"), dict) else {}
                    args = state.get("input") if isinstance(state.get("input"), dict) else {}
                    tool_call = {
                        "kind": "call_tool",
                        "tool_name": tool_name,
                        "args": dict(args),
                    }
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                content = f"{content}\n{text.strip()}".strip()
                continue
            part_content = part.get("content")
            if isinstance(part_content, str) and part_content.strip():
                content = f"{content}\n{part_content.strip()}".strip()
    if not content:
        content = (_extract_session_message_content(body) or _extract_message_content(body) or "").strip()
    return content, tool_call


def _extract_canonical_from_text_content(
    content: str,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    parsed = _try_parse_json_object(content)
    if not isinstance(parsed, dict):
        return None, None, None

    canonical_tool_call = parsed.get("tool_call")
    tool_call: dict[str, Any] | None = canonical_tool_call if isinstance(canonical_tool_call, dict) else None

    planner_intent_raw = parsed.get("planner_intent")
    planner_intent: str | None = None
    if isinstance(planner_intent_raw, str):
        text = planner_intent_raw.strip()
        if text:
            planner_intent = text[:160]

    parsed_content = parsed.get("content")
    if isinstance(parsed_content, str):
        return tool_call, planner_intent, parsed_content.strip()
    if tool_call is not None or planner_intent is not None:
        return tool_call, planner_intent, ""
    return None, None, None


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
        # Try raw decoder from each opening brace to tolerate pre/post text wrappers.
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
        error_prefix="opencode_complete_with_tools_non_canonical",
    )
