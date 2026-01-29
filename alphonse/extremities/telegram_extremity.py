"""Telegram extremity for Alphonse."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from alphonse.agent.runtime import get_runtime
from alphonse.cognition.providers.ollama import OllamaClient
from alphonse.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.senses.bus import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: str
    allowed_chat_ids: set[int] | None
    poll_interval_sec: float


@dataclass(frozen=True)
class IncomingMessage:
    text: str
    chat_id: int
    user_id: str | None
    timestamp: float
    channel: str = "telegram"
    message_id: int | None = None


@dataclass(frozen=True)
class SkillResult:
    skill: str
    args: dict[str, Any]
    confidence: float
    needs_clarification: bool
    clarifying_question: str | None


def build_telegram_extremity_from_env() -> "TelegramExtremity | None":
    enabled = _env_flag("ALPHONSE_ENABLE_TELEGRAM")
    if not enabled:
        return None

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required when Telegram is enabled")

    allowed_chat_ids = _parse_allowed_chat_ids(
        os.getenv("TELEGRAM_ALLOWED_CHAT_IDS"),
        os.getenv("TELEGRAM_ALLOWED_CHAT_ID"),
    )
    poll_interval = _parse_float(os.getenv("TELEGRAM_POLL_INTERVAL_SEC"), default=1.0)

    settings = TelegramSettings(
        bot_token=bot_token,
        allowed_chat_ids=allowed_chat_ids,
        poll_interval_sec=poll_interval,
    )
    return TelegramExtremity(settings)


class TelegramExtremity:
    def __init__(self, settings: TelegramSettings) -> None:
        self._settings = settings
        self._running = False
        adapter_config: dict[str, Any] = {
            "bot_token": settings.bot_token,
            "poll_interval_sec": settings.poll_interval_sec,
        }
        if settings.allowed_chat_ids is not None:
            adapter_config["allowed_chat_ids"] = list(settings.allowed_chat_ids)
        self._adapter = TelegramAdapter(adapter_config)

    def start(self) -> None:
        if self._running:
            return
        self._adapter.on_signal(self._on_signal)
        self._adapter.start()
        self._running = True
        logger.info("Telegram extremity started")

    def stop(self) -> None:
        if not self._running:
            return
        self._adapter.stop()
        self._running = False
        logger.info("Telegram extremity stopped")

    def _on_signal(self, signal: Signal) -> None:
        if signal.type != "external.telegram.message":
            return
        payload = signal.payload or {}
        chat_id = payload.get("chat_id")
        if chat_id is None:
            logger.warning("Telegram message missing chat_id")
            return
        if self._settings.allowed_chat_ids and int(chat_id) not in self._settings.allowed_chat_ids:
            logger.info("Telegram message ignored from chat_id=%s", chat_id)
            return

        text = str(payload.get("text", "")).strip()
        if not text:
            return

        message = IncomingMessage(
            text=text,
            chat_id=int(chat_id),
            user_id=_as_user_id(payload.get("from_user")),
            timestamp=time.time(),
            message_id=payload.get("message_id"),
        )
        try:
            self._handle_message(message)
        except Exception as exc:
            logger.exception("Telegram message handling failed: %s", exc)

    def _handle_message(self, message: IncomingMessage) -> None:
        result = self._route_message(message)
        response_text = self._format_result(result, message)
        if not response_text:
            return
        self._send_message(message.chat_id, response_text)

    def _route_message(self, message: IncomingMessage) -> SkillResult:
        text_lower = message.text.lower()
        if "status" in text_lower:
            return SkillResult(
                skill="system.status",
                args={},
                confidence=1.0,
                needs_clarification=False,
                clarifying_question=None,
            )
        if "joke" in text_lower:
            return SkillResult(
                skill="system.joke",
                args={},
                confidence=1.0,
                needs_clarification=False,
                clarifying_question=None,
            )

        interpretation = self._interpret_message(message.text)
        if interpretation is None:
            return SkillResult(
                skill="conversation.echo",
                args={},
                confidence=0.0,
                needs_clarification=False,
                clarifying_question=None,
            )

        return SkillResult(
            skill=str(interpretation.get("skill") or "conversation.echo"),
            args=dict(interpretation.get("args") or {}),
            confidence=float(interpretation.get("confidence") or 0.0),
            needs_clarification=bool(interpretation.get("needs_clarification", False)),
            clarifying_question=interpretation.get("clarifying_question"),
        )

    def _format_result(self, result: SkillResult, message: IncomingMessage) -> str:
        if result.needs_clarification and result.clarifying_question:
            return result.clarifying_question

        if result.skill == "system.status":
            return self._llm_status()
        if result.skill == "system.joke":
            return self._llm_joke()
        if result.skill == "conversation.echo":
            return f"Echo: {message.text}"

        return f"Echo: {message.text}"

    def _interpret_message(self, text: str) -> dict[str, Any] | None:
        client = _build_ollama_client()
        system_prompt = (
            "You are Alphonse, a calm and restrained domestic presence. "
            "Classify the message into a skill and return JSON only. "
            "Do not include code fences or extra text.\n\n"
            "Schema:\n"
            '{"skill": "system.status", "args": {}, "confidence": 0.82, '
            '"needs_clarification": false, "clarifying_question": null}\n\n'
            "Skills:\n"
            "- system.status: household status request\n"
            "- system.joke: user wants a gentle joke\n"
            "- conversation.echo: fallback for simple acknowledgement\n"
        )
        user_prompt = f"Message: {text}" 
        try:
            content = client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = _parse_json(str(content))
            if not parsed:
                logger.warning("Telegram LLM returned invalid JSON")
            return parsed
        except Exception as exc:
            logger.warning("Telegram LLM interpretation failed: %s", exc)
            return None

    def _llm_status(self) -> str:
        runtime = get_runtime().snapshot()
        uptime = max(0, int(float(runtime.get("uptime_seconds", 0))))
        last_signal = runtime.get("last_signal", {}) or {}
        last_signal_type = last_signal.get("type")
        last_signal_at = last_signal.get("ts")
        system_prompt = (
            "You are Alphonse, a calm and restrained domestic presence.\n"
            "Summarize the current status if requested.\n"
            "Do not suggest actions.\n"
            "Do not speculate.\n"
            "Keep it under 2 sentences."
        )
        prompt = (
            f"{system_prompt}\n\n"
            "Runtime snapshot:\n"
            f"- Uptime: {uptime}s\n"
            f"- Heartbeat ticks: {runtime.get('tick_count', 0)}\n"
            f"- Last signal type: {last_signal_type}\n"
            f"- Last signal at: {last_signal_at}\n"
        )
        try:
            client = _build_ollama_client()
            content = client.complete(system_prompt=system_prompt, user_prompt=prompt)
            if content:
                return str(content).strip()
        except Exception as exc:
            logger.warning("Ollama status call failed: %s", exc)

        return (
            "Alphonse status: "
            f"uptime {uptime}s, "
            f"last signal {last_signal_type} at {last_signal_at}."
        )

    def _llm_joke(self) -> str:
        system_prompt = (
            "You are Alphonse, a calm and restrained domestic presence.\n"
            "Tell a short, gentle joke in one sentence.\n"
            "Avoid sarcasm or insults.\n"
            "Keep it under 2 sentences."
        )
        prompt = "Please provide a short, kind joke."
        try:
            client = _build_ollama_client()
            content = client.complete(system_prompt=system_prompt, user_prompt=prompt)
            if content:
                return str(content).strip()
        except Exception as exc:
            logger.warning("Ollama joke call failed: %s", exc)

        return "Here is a gentle joke: Why did the scarecrow get promoted? Because he was outstanding in his field."

    def _send_message(self, chat_id: int, text: str) -> None:
        self._adapter.handle_action(
            {
                "type": "send_message",
                "payload": {
                    "chat_id": chat_id,
                    "text": text,
                },
                "target_integration_id": "telegram",
            }
        )


def _build_ollama_client() -> OllamaClient:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct")
    timeout_seconds = _parse_float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS"), default=240.0)
    return OllamaClient(
        base_url=base_url,
        model=model,
        timeout=timeout_seconds,
    )


def _parse_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _parse_allowed_chat_ids(primary: str | None, fallback: str | None) -> set[int] | None:
    raw = primary or fallback
    if not raw:
        return None
    ids: set[int] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            ids.add(int(entry))
        except ValueError:
            logger.warning("Invalid telegram chat id: %s", entry)
    return ids or None


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _as_user_id(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)
