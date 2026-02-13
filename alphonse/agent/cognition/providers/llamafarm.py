from __future__ import annotations

import os

import requests


class LlamaFarmClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8002/v1",
        model: str = "mistral:7b-instruct",
        api_key_env: str = "LLAMAFARM_API_KEY",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LlamaFarm response missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("LlamaFarm response choice malformed")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("LlamaFarm response missing message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LlamaFarm response missing assistant content")
        return content
