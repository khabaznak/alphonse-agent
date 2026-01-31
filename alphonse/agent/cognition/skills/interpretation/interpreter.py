from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from alphonse.agent.cognition.skills.interpretation.models import MessageEvent, RoutingDecision
from alphonse.agent.cognition.skills.interpretation.registry import SkillRegistry, _normalize_command


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True)
class InterpreterConfig:
    clarification_prompt: str = "Could you clarify what you need?"


class MessageInterpreter:
    def __init__(
        self,
        registry: SkillRegistry,
        llm_client: LLMClient,
        config: InterpreterConfig | None = None,
    ) -> None:
        self._registry = registry
        self._llm_client = llm_client
        self._config = config or InterpreterConfig()

    def interpret(self, message: MessageEvent) -> RoutingDecision:
        normalized = _normalize_command(message.text)
        if not normalized:
            return self._clarify("I did not catch that. Could you rephrase?")

        reminder_decision = self._maybe_schedule_reminder(message)
        if reminder_decision:
            return reminder_decision

        deterministic = self._registry.match_alias(normalized)
        if deterministic:
            args = self._extract_args(message)
            if not self._validate_args(deterministic.key, args):
                return self._clarify(self._config.clarification_prompt)
            return RoutingDecision(
                skill=deterministic.key,
                args=args,
                confidence=1.0,
                needs_clarification=False,
                clarifying_question=None,
                path="deterministic",
            )

        llm_result = self._interpret_with_llm(message)
        if llm_result is None:
            return self._clarify(self._config.clarification_prompt)

        skill = str(llm_result.get("skill") or "").strip()
        if not skill or not self._registry.get_skill(skill):
            return self._clarify(self._config.clarification_prompt)

        args = llm_result.get("args") or {}
        if not isinstance(args, dict):
            return self._clarify(self._config.clarification_prompt)

        if not self._validate_args(skill, args):
            return self._clarify(self._config.clarification_prompt)

        needs_clarification = bool(llm_result.get("needs_clarification", False))
        clarifying_question = llm_result.get("clarifying_question")
        if needs_clarification and not clarifying_question:
            clarifying_question = self._config.clarification_prompt

        return RoutingDecision(
            skill=skill,
            args=args,
            confidence=float(llm_result.get("confidence", 0.0)),
            needs_clarification=needs_clarification,
            clarifying_question=clarifying_question,
            path="llm",
        )

    def _interpret_with_llm(self, message: MessageEvent) -> dict[str, Any] | None:
        system_prompt = self._build_system_prompt()
        user_prompt = f"Message: {message.text}"
        try:
            content = self._llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception:
            return None
        return _parse_json(str(content))

    def _build_system_prompt(self) -> str:
        skills = self._registry.list_skills()
        skill_lines = "\n".join(
            _format_skill_line(skill.key, skill.description, skill.arg_schema) for skill in skills
        )
        return (
            "You are Alphonse, a calm and restrained domestic presence. "
            "Choose exactly one skill from the list and respond with JSON only. "
            "Do not include code fences or extra text.\n\n"
            "Schema:\n"
            '{"skill": "system.status", "args": {}, "confidence": 0.82, '
            '"needs_clarification": false, "clarifying_question": null}\n\n'
            "Skills:\n"
            f"{skill_lines}\n\n"
            "If you are unsure, set needs_clarification=true and provide a clarifying_question."
        )

    def _validate_args(self, skill_key: str, args: dict[str, Any]) -> bool:
        skill = self._registry.get_skill(skill_key)
        if not skill or not skill.arg_schema:
            return True
        allowed_keys = set(skill.arg_schema.keys())
        return set(args.keys()).issubset(allowed_keys)

    def _extract_args(self, message: MessageEvent) -> dict[str, Any]:
        args = message.metadata.get("args")
        if isinstance(args, dict):
            return args
        return {}

    def _maybe_schedule_reminder(self, message: MessageEvent) -> RoutingDecision | None:
        text = message.text.lower()
        if "remind" not in text and "reminder" not in text and "schedule" not in text:
            return None
        return RoutingDecision(
            skill="schedule.timed_signal",
            args={"signal_type": "reminder"},
            confidence=0.8,
            needs_clarification=False,
            clarifying_question=None,
            path="deterministic",
        )

    def _clarify(self, prompt: str) -> RoutingDecision:
        return RoutingDecision(
            skill="conversation.echo",
            args={},
            confidence=0.0,
            needs_clarification=True,
            clarifying_question=prompt,
            path="llm",
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


def _format_skill_line(key: str, description: str, arg_schema: dict[str, Any] | None) -> str:
    if not arg_schema:
        return f"- {key}: {description}"
    args = ", ".join(sorted(arg_schema.keys()))
    return f"- {key}: {description} (args: {args})"
