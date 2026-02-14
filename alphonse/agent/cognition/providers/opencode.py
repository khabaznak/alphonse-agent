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
        model: str = "ollama/mistral:7b-instruct",
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

        response = requests.post(
            f"{self.base_url}{self.chat_path}",
            headers=headers,
            auth=auth,
            json=payload,
            timeout=self.timeout,
        )
        _raise_for_status_with_body(response, "OpenCode chat completion failed")
        body = _read_json_response(response)
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
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
            "stream": False,
        }
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
        response = requests.post(
            f"{self.base_url}{self.chat_path}",
            headers=headers,
            auth=auth,
            json=payload,
            timeout=self.timeout,
        )
        _raise_for_status_with_body(response, "OpenCode tool completion failed")
        body = _read_json_response(response)
        content, tool_calls, assistant_message = _extract_tool_call_payload(body)
        return {
            "content": content,
            "tool_calls": tool_calls,
            "assistant_message": assistant_message,
        }

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
