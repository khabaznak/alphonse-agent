import os
import json
from typing import Any

import requests


class OpenAIClient:
    def __init__(
        self,
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        timeout=60,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.supports_tool_calls = True

    def complete(self, system_prompt, user_prompt):
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in {self.api_key_env}.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in {self.api_key_env}.")
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            return {"content": "", "tool_calls": []}
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return {"content": "", "tool_calls": []}
        content = message.get("content")
        if not isinstance(content, str):
            content = ""
        raw_tool_calls = message.get("tool_calls")
        tool_calls: list[dict[str, Any]] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        if isinstance(raw_tool_calls, list):
            for item in raw_tool_calls:
                if not isinstance(item, dict):
                    continue
                function_obj = item.get("function")
                if not isinstance(function_obj, dict):
                    continue
                name = str(function_obj.get("name") or "").strip()
                arguments_raw = function_obj.get("arguments")
                args: dict[str, Any] = {}
                if isinstance(arguments_raw, str) and arguments_raw.strip():
                    try:
                        parsed = json.loads(arguments_raw)
                        if isinstance(parsed, dict):
                            args = parsed
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(arguments_raw, dict):
                    args = arguments_raw
                if not name:
                    continue
                assistant_tool_calls.append(
                    {
                        "id": str(item.get("id") or "").strip(),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": (
                                arguments_raw
                                if isinstance(arguments_raw, str)
                                else json.dumps(args, ensure_ascii=False)
                            ),
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
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls
        return {"content": content, "tool_calls": tool_calls, "assistant_message": assistant_message}
