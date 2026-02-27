from __future__ import annotations

import os
import json
from typing import Any

import requests


class OpenCodeClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:4096",
        model: str = "openai/gpt-5.1-codex",
        timeout: float = 120.0,
        chat_path: str = "/v1/chat/completions",
        api_key_env: str = "OPENCODE_API_KEY",
        username_env: str = "OPENCODE_SERVER_USERNAME",
        password_env: str = "OPENCODE_SERVER_PASSWORD",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
        self.api_key_env = api_key_env
        self.username_env = username_env
        self.password_env = password_env
        self._session_id: str | None = None
        self.supports_tool_calls = True
        self.tool_result_message_style = "openai"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        force_session = str(os.getenv("OPENCODE_FORCE_SESSION_API", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if force_session:
            return self._complete_via_session_api(system_prompt, user_prompt)
        # Prefer OpenAI-compatible route when available.
        try:
            return self._complete_via_chat_completions(system_prompt, user_prompt)
        except Exception:
            # Fall back to the native OpenCode session/message API.
            return self._complete_via_session_api(system_prompt, user_prompt)

    def _complete_via_chat_completions(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        body = self._post_chat_completion(payload)
        content = _extract_message_content(body)
        if content is None:
            raise ValueError("OpenCode response missing assistant content")
        return content

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        mode = str(os.getenv("OPENCODE_TOOL_CALL_MODE", "session")).strip().lower()
        if mode not in {"session", "chat"}:
            mode = "session"

        if mode == "session":
            return self._complete_with_tools_via_session_api(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

        return self._complete_with_tools_via_chat_completions(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

    def _complete_with_tools_via_chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
            "stream": False,
        }
        body = self._post_chat_completion(payload)
        content, tool_calls, assistant_message = _extract_tool_call_payload(body)
        return {
            "content": content,
            "tool_calls": tool_calls,
            "assistant_message": assistant_message,
        }

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
        rendered = _render_tool_call_markdown_prompt(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        model_payloads: list[Any] = [_model_for_session_payload(self.model), self.model]
        payloads = [
            {"model": m, "parts": [{"type": "text", "text": rendered}]}
            for m in model_payloads
            if m
        ]
        payloads.append({"parts": [{"type": "text", "text": rendered}]})

        failures: list[str] = []
        for idx, payload in enumerate(payloads):
            try:
                response = requests.post(
                    f"{self.base_url}/session/{session_id}/message",
                    headers=headers,
                    auth=auth,
                    json=payload,
                    timeout=tool_timeout,
                )
                _raise_for_status_with_body(
                    response,
                    f"OpenCode session tool-call message failed (payload_variant={idx})",
                )
                body = _read_json_response(response)
                content, tool_calls, assistant_message = _extract_session_tool_payload(body)
                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "assistant_message": assistant_message,
                }
            except Exception as exc:
                failures.append(str(exc))
        raise ValueError(
            "OpenCode session tool-call loop failed for all payload variants: "
            + " | ".join(failures[:4])
        )

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        auth = None
        api_key = os.getenv(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            username = os.getenv(self.username_env, "opencode")
            password = os.getenv(self.password_env)
            if password:
                auth = (username, password)
        paths = [self.chat_path]
        if self.chat_path.startswith("/v1/"):
            paths.append(f"/api{self.chat_path}")
        elif self.chat_path.startswith("/api/v1/"):
            paths.append(self.chat_path.replace("/api", "", 1))
        errors: list[str] = []
        for path in paths:
            try:
                response = requests.post(
                    f"{self.base_url}{path}",
                    headers=headers,
                    auth=auth,
                    json=payload,
                    timeout=self.timeout,
                )
                _raise_for_status_with_body(response, f"OpenCode chat completion failed path={path}")
                return _read_json_response(response)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
        raise ValueError(
            "OpenCode chat completion failed for all configured paths. "
            f"base_url={self.base_url} chat_path={self.chat_path} "
            "Hint: point OPENCODE_BASE_URL to the JSON API host (not the web UI), "
            "or set OPENCODE_CHAT_COMPLETIONS_PATH correctly. "
            f"errors={' | '.join(errors[:3])}"
        )

    def _complete_via_session_api(self, system_prompt: str, user_prompt: str) -> str:
        headers, auth = _build_auth(self.api_key_env, self.username_env, self.password_env)
        session_id = self._ensure_session(headers=headers, auth=auth)
        composite_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}".strip()
        payload_candidates = [
            {
                "model": self.model,
                "system": system_prompt,
                "parts": [{"type": "text", "text": user_prompt}],
            },
            {
                "parts": [{"type": "text", "text": composite_prompt}],
            },
            {
                "parts": [{"text": composite_prompt}],
            },
            {
                "message": {"role": "user", "content": composite_prompt},
            },
        ]
        failures: list[str] = []
        for idx, payload in enumerate(payload_candidates):
            try:
                response = requests.post(
                    f"{self.base_url}/session/{session_id}/message",
                    headers=headers,
                    auth=auth,
                    json=payload,
                    timeout=self.timeout,
                )
                _raise_for_status_with_body(
                    response,
                    f"OpenCode session message failed (payload_variant={idx})",
                )
                body = _read_json_response(response)
                content = _extract_session_message_content(body) or _extract_message_content(body)
                if content:
                    return content
                failures.append(f"payload_variant={idx} returned no content")
            except Exception as exc:
                failures.append(str(exc))
        raise ValueError(
            "OpenCode session message failed for all payload variants: "
            + " | ".join(failures[:4])
        )

    def _ensure_session(self, *, headers: dict[str, str], auth: tuple[str, str] | None) -> str:
        if self._session_id:
            return self._session_id
        session_payloads = [
            {"model": self.model},
            {},
        ]
        last_error: Exception | None = None
        for payload in session_payloads:
            try:
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
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise ValueError("OpenCode session creation failed")


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
        parsed = response.json()
    except ValueError:
        text = (response.text or "").strip()
        if text.startswith("data:"):
            # Some servers can still return SSE-formatted chunks; use the first JSON chunk.
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


def _extract_tool_call_payload(body: Any) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    content = _extract_message_content(body) or ""
    tool_calls: list[dict[str, Any]] = []
    assistant_tool_calls: list[dict[str, Any]] = []

    if isinstance(body, dict):
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else {}
            raw_calls = message.get("tool_calls") if isinstance(message, dict) else None
            if isinstance(raw_calls, list):
                for item in raw_calls:
                    if not isinstance(item, dict):
                        continue
                    function_obj = item.get("function")
                    if not isinstance(function_obj, dict):
                        continue
                    name = str(function_obj.get("name") or "").strip()
                    if not name:
                        continue
                    raw_args = function_obj.get("arguments")
                    args: dict[str, Any] = {}
                    if isinstance(raw_args, str) and raw_args.strip():
                        try:
                            parsed = json.loads(raw_args)
                            if isinstance(parsed, dict):
                                args = parsed
                        except json.JSONDecodeError:
                            args = {}
                    elif isinstance(raw_args, dict):
                        args = raw_args
                    assistant_tool_calls.append(
                        {
                            "id": str(item.get("id") or "").strip(),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": raw_args if isinstance(raw_args, str) else json.dumps(args, ensure_ascii=False),
                            },
                        }
                    )
                    tool_calls.append(
                        {
                            "id": str(item.get("id") or "").strip(),
                            "name": name,
                            "arguments": args,
                        }
                    )

    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
    if assistant_tool_calls:
        assistant_message["tool_calls"] = assistant_tool_calls
    return content, tool_calls, assistant_message


def _render_tool_call_markdown_prompt(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str,
) -> str:
    lines: list[str] = []
    lines.append("# Tool Call Task")
    lines.append("")
    lines.append("You are in a tool-calling loop.")
    lines.append("Return ONLY valid JSON.")
    lines.append("")
    lines.append("## Output JSON Contract")
    lines.append('{"content":"string","tool_calls":[{"id":"optional","name":"toolName","arguments":{}}]}')
    lines.append("")
    lines.append("Rules:")
    lines.append("- Use `tool_calls` when a tool is needed next.")
    lines.append("- Use empty `tool_calls` when you can answer directly.")
    lines.append("- `arguments` must be an object.")
    lines.append("- No markdown, no prose outside JSON.")
    lines.append("")
    lines.append(f"## Tool Choice")
    lines.append(f"- {tool_choice}")
    lines.append("")
    lines.append("## Tools")
    if not tools:
        lines.append("- (none)")
    else:
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            description = str(fn.get("description") or "").strip()
            parameters = fn.get("parameters")
            lines.append(f"- `{name}`")
            if description:
                lines.append(f"  - {description}")
            if isinstance(parameters, dict):
                lines.append("  - parameters schema:")
                lines.append("```json")
                lines.append(json.dumps(parameters, ensure_ascii=False))
                lines.append("```")
    lines.append("")
    lines.append("## Conversation")
    if not messages:
        lines.append("- (empty)")
    else:
        for message in messages:
            role = str(message.get("role") or "unknown")
            content = str(message.get("content") or "").strip()
            lines.append(f"### {role}")
            if content:
                lines.append(content)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                lines.append("tool_calls:")
                lines.append("```json")
                lines.append(json.dumps(tool_calls, ensure_ascii=False))
                lines.append("```")
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                lines.append(f"tool_call_id: `{tool_call_id}`")
            tool_name = message.get("tool_name")
            if isinstance(tool_name, str) and tool_name.strip():
                lines.append(f"tool_name: `{tool_name}`")
            lines.append("")
    return "\n".join(lines).strip()


def _model_for_session_payload(model_ref: str) -> dict[str, str] | None:
    value = str(model_ref or "").strip()
    if not value or "/" not in value:
        return None
    provider, model_id = value.split("/", 1)
    provider = provider.strip()
    model_id = model_id.strip()
    if not provider or not model_id:
        return None
    return {"providerID": provider, "modelID": model_id}


def _extract_session_tool_payload(body: Any) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    content = ""
    tool_calls: list[dict[str, Any]] = []
    assistant_tool_calls: list[dict[str, Any]] = []
    parts = body.get("parts") if isinstance(body, dict) else None
    if isinstance(parts, list):
        for idx, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip().lower()
            if part_type == "tool":
                tool_name = str(part.get("tool") or "").strip()
                if not tool_name:
                    continue
                call_id = str(part.get("callID") or part.get("call_id") or f"call-{idx}").strip()
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                args = state.get("input") if isinstance(state.get("input"), dict) else {}
                tool_calls.append({"id": call_id, "name": tool_name, "arguments": args})
                assistant_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
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

    if not tool_calls and content:
        parsed = _try_parse_json_object(content)
        if isinstance(parsed, dict):
            maybe_calls = parsed.get("tool_calls")
            maybe_content = parsed.get("content")
            if isinstance(maybe_calls, list):
                for idx, call in enumerate(maybe_calls):
                    if not isinstance(call, dict):
                        continue
                    name = str(call.get("name") or "").strip()
                    if not name:
                        continue
                    call_id = str(call.get("id") or f"call-{idx}").strip()
                    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                    tool_calls.append({"id": call_id, "name": name, "arguments": args})
                    assistant_tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    )
            if isinstance(maybe_content, str):
                content = maybe_content.strip()

    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
    if assistant_tool_calls:
        assistant_message["tool_calls"] = assistant_tool_calls
    return content, tool_calls, assistant_message


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if "```json" in raw:
        start = raw.find("```json")
        end = raw.find("```", start + 7)
        if start >= 0 and end > start:
            candidates.append(raw[start + 7 : end].strip())
    if "```" in raw:
        first = raw.find("```")
        second = raw.find("```", first + 3)
        if first >= 0 and second > first:
            block = raw[first + 3 : second].strip()
            if block:
                candidates.append(block)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
