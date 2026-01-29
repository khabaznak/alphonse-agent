from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from alphonse.agent.runtime import get_runtime
from alphonse.cognition.providers.ollama import OllamaClient
from alphonse.interpretation.models import MessageEvent, RoutingDecision
from alphonse.interpretation.registry import SkillRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillExecutor:
    registry: SkillRegistry
    llm_client: OllamaClient
    clarification_prompt: str = "Could you clarify what you need?"

    def respond(self, decision: RoutingDecision, message: MessageEvent) -> str:
        if decision.needs_clarification:
            return decision.clarifying_question or self.clarification_prompt

        if decision.skill == "system.status":
            return self._status_response()
        if decision.skill == "system.joke":
            return self._joke_response()
        if decision.skill == "system.help":
            return self._format_help()
        if decision.skill == "conversation.echo":
            return f"Echo: {message.text}"

        return f"Echo: {message.text}"

    def _status_response(self) -> str:
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
            content = self.llm_client.complete(system_prompt=system_prompt, user_prompt=prompt)
            if content:
                return str(content).strip()
        except Exception as exc:
            logger.warning("Ollama status call failed: %s", exc)

        return (
            "Alphonse status: "
            f"uptime {uptime}s, "
            f"last signal {last_signal_type} at {last_signal_at}."
        )

    def _joke_response(self) -> str:
        system_prompt = (
            "You are Alphonse, a calm and restrained domestic presence.\n"
            "Tell a short, gentle joke in one sentence.\n"
            "Avoid sarcasm or insults.\n"
            "Keep it under 2 sentences."
        )
        prompt = "Please provide a short, kind joke."
        try:
            content = self.llm_client.complete(system_prompt=system_prompt, user_prompt=prompt)
            if content:
                return str(content).strip()
        except Exception as exc:
            logger.warning("Ollama joke call failed: %s", exc)

        return "Here is a gentle joke: Why did the scarecrow get promoted? Because he was outstanding in his field."

    def _format_help(self) -> str:
        lines = ["Available commands:"]
        for skill in sorted(self.registry.list_skills(), key=lambda item: item.key):
            if skill.key == "conversation.echo":
                continue
            aliases = [alias for alias in skill.aliases if alias and not alias.startswith("/")]
            alias_text = ", ".join(sorted(set(aliases)))
            if alias_text:
                lines.append(f"- {skill.key}: {alias_text}")
            else:
                lines.append(f"- {skill.key}")
        return "\n".join(lines)


def build_ollama_client() -> OllamaClient:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct")
    timeout_seconds = _parse_float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS"), default=240.0)
    return OllamaClient(
        base_url=base_url,
        model=model,
        timeout=timeout_seconds,
    )


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
