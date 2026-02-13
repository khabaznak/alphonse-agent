from __future__ import annotations

import os
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

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
        response.raise_for_status()
        body = response.json()
        content = _extract_message_content(body)
        if content is None:
            raise ValueError("OpenCode response missing assistant content")
        return content


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
