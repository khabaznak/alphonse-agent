import json
import os
from urllib import request
from alphonse.agent.cognition.provider_selector import build_provider_client
from alphonse.agent.cognition.reasoner import build_system_prompt
from alphonse.config import load_alphonse_config


STATUS_SYSTEM_PROMPT = (
    "You are Alphonse, a calm and restrained domestic presence.\n\n"
    "Given the following system snapshot, summarize the current state calmly in one sentence.\n"
    "Do not suggest actions.\n"
    "Do not speculate.\n"
    "Keep it under 2 sentences."
)


def _fetch_runtime_status() -> dict:
    base_url = os.getenv("ATRIUM_HTTP_BASE_URL", "http://localhost:8000")
    url = f"{base_url.rstrip('/')}/agent/status"
    try:
        with request.urlopen(url, timeout=2) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except Exception:
        return {"runtime": None, "error": "status_unavailable"}


def reason_about_status():
    snapshot = _fetch_runtime_status()
    config = load_alphonse_config()
    client = build_provider_client(config)

    system_prompt = _build_system_prompt(config)
    message = client.complete(
        system_prompt=system_prompt,
        user_prompt=_build_user_prompt(snapshot),
    )

    return message, snapshot


def _build_system_prompt(config: dict) -> str:
    mode = str(config.get("mode", "test")).lower()
    if mode == "test":
        return build_system_prompt()

    return STATUS_SYSTEM_PROMPT


def _build_user_prompt(snapshot: dict) -> str:
    return f"{STATUS_SYSTEM_PROMPT}\n\n{snapshot}"
