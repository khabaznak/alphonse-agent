import json
from typing import Any

import requests


class OllamaClient:
    def __init__(
        self,
        base_url="http://localhost:11434",
        model="mistral:7b-instruct",
        timeout=120,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.supports_tool_calls = True
        self.tool_result_message_style = "ollama"

    def complete(self, system_prompt, user_prompt):
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
    ) -> dict[str, Any]:
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
        tool_calls: list[dict[str, Any]] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_tool_calls):
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
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str) and raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    args = {}
            call_id = str(item.get("id") or f"ollama-call-{idx}").strip()
            tool_calls.append({"id": call_id, "name": name, "arguments": args})
            assistant_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": raw_args if isinstance(raw_args, str) else json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls
        return {"content": content, "tool_calls": tool_calls, "assistant_message": assistant_message}
