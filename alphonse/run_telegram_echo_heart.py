"""Echo heart runner for Telegram integration."""

from __future__ import annotations

import logging
import os
from alphonse.cognition.providers.ollama import OllamaClient
import time
from typing import Any
from pathlib import Path

from dotenv import load_dotenv

from alphonse.extremities.interfaces.integrations.loader import IntegrationLoader
from alphonse.senses.bus import Bus, Signal


def llm_status(state: dict[str, Any]) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct")
    timeout_raw = os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "240")
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 240.0
    uptime = max(0, int(time.time() - float(state.get("boot_time", time.time()))))
    signals_seen = state.get("signals_seen", 0)
    last_signal_type = state.get("last_signal_type")
    last_signal_at = state.get("last_signal_at")

    system_prompt = (
        "You are Alphonse, a calm and restrained domestic presence.\n"
        "Summarize the current status if requested\n"
        "Do not suggest actions.\n"
        "Do not speculate.\n"
        "Keep it under 2 sentences."
    )
    prompt = (
        f"{system_prompt}\n\n"
        "Runtime snapshot:\n"
        f"- Uptime: {uptime}s\n"
        f"- Signals seen: {signals_seen}\n"
        f"- Last signal type: {last_signal_type}\n"
        f"- Last signal at: {last_signal_at}\n"
    )

    try:
        client = OllamaClient(
            base_url=base_url,
            model=model,
            timeout=timeout_seconds,
        )
        content = client.complete(system_prompt=system_prompt, user_prompt=prompt)
        if content:
            return str(content).strip()
    except Exception as exc:
        logging.warning("Ollama call failed: %s", exc)

    return (
        "Alphonse status: "
        f"uptime {uptime}s, "
        f"signals seen {signals_seen}, "
        f"last signal {last_signal_type} at {last_signal_at}."
    )


def llm_joke() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct")
    timeout_raw = os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "240")
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 240.0

    system_prompt = (
        "You are Alphonse, a calm and restrained domestic presence.\n"
        "Tell a short, gentle joke in one sentence.\n"
        "Avoid sarcasm or insults.\n"
        "Keep it under 2 sentences."
    )
    prompt = "Please provide a short, kind joke."

    try:
        client = OllamaClient(
            base_url=base_url,
            model=model,
            timeout=timeout_seconds,
        )
        content = client.complete(system_prompt=system_prompt, user_prompt=prompt)
        if content:
            return str(content).strip()
    except Exception as exc:
        logging.warning("Ollama call failed: %s", exc)

    return "Here is a gentle joke: Why did the scarecrow get promoted? Because he was outstanding in his field."


def heart_step(signal: Signal, state: dict[str, Any]) -> dict[str, object] | None:
    if signal.type != "external.telegram.message":
        return None
    payload = signal.payload or {}
    text = str(payload.get("text", "")).strip().lower()
    if "status" in text:
        logging.info("Status command detected")
        status_text = llm_status(state)
        return {
            "type": "send_message",
            "payload": {
                "chat_id": payload.get("chat_id"),
                "text": status_text,
            },
            "target_integration_id": "telegram",
        }
    if "joke" in text:
        logging.info("Joke command detected")
        joke_text = llm_joke()
        return {
            "type": "send_message",
            "payload": {
                "chat_id": payload.get("chat_id"),
                "text": joke_text,
            },
            "target_integration_id": "telegram",
        }
    return {
        "type": "send_message",
        "payload": {
            "chat_id": payload.get("chat_id"),
            "text": f"Echo: {payload.get('text', '')}",
        },
        "target_integration_id": "telegram",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    env_path = Path(__file__).resolve().parent / "agent" / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    allowed_chat_id = os.getenv("TELEGRAM_ALLOWED_CHAT_ID")
    allowed_chat_ids = [int(allowed_chat_id)] if allowed_chat_id else None

    bus = Bus()
    config = {
        "integrations": [
            {
                "name": "telegram",
                "enabled": True,
                "module": "alphonse.extremities.interfaces.integrations.telegram.telegram_adapter",
                "class": "TelegramAdapter",
                "config": {
                    "bot_token": token,
                    "allowed_chat_ids": allowed_chat_ids,
                },
            }
        ]
    }

    loader = IntegrationLoader(config, bus)
    registry = loader.load_all()
    loader.start_all(registry)

    state: dict[str, Any] = {
        "boot_time": time.time(),
        "signals_seen": 0,
        "last_signal_type": None,
        "last_signal_at": None,
    }

    try:
        while True:
            signal = bus.get(timeout=None)
            if signal is None:
                continue
            state["signals_seen"] = int(state.get("signals_seen", 0)) + 1
            state["last_signal_type"] = signal.type
            state["last_signal_at"] = time.time()
            logging.info(
                "Received signal: type=%s source=%s payload=%s id=%s",
                signal.type,
                signal.source,
                signal.payload,
                signal.id,
            )
            action = heart_step(signal, state)
            if action is None:
                continue
            adapter = registry.get("telegram")
            if adapter is None:
                logging.error("Telegram adapter not found")
                continue
            adapter.handle_action(action)
            logging.info("Delivered action to telegram adapter")
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        loader.stop_all(registry)


if __name__ == "__main__":
    main()
