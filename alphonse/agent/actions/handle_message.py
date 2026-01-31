from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.skills.interpretation.interpreter import MessageInterpreter
from alphonse.agent.cognition.skills.interpretation.models import MessageEvent
from alphonse.agent.cognition.skills.interpretation.registry import build_default_registry
from alphonse.agent.cognition.skills.interpretation.skills import SkillExecutor, build_ollama_client


class HandleMessageAction(Action):
    key = "handle_message"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        correlation_id = getattr(signal, "correlation_id", None) if signal else None

        text = str(payload.get("text", "")).strip()
        chat_id = payload.get("chat_id")
        user_id = payload.get("from_user")
        user_name = payload.get("from_user_name") or user_id
        origin = payload.get("origin") or "telegram"

        registry = build_default_registry()
        llm_client = build_ollama_client()
        interpreter = MessageInterpreter(registry, llm_client)
        executor = SkillExecutor(registry, llm_client)

        metadata = {
            "chat_id": chat_id,
            "target": chat_id,
            "user_name": user_name,
        }
        args = payload.get("args")
        if isinstance(args, dict):
            metadata["args"] = args

        event = MessageEvent(
            text=text,
            user_id=str(user_id) if user_id is not None else None,
            channel=str(origin),
            timestamp=payload.get("timestamp") or 0.0,
            metadata=metadata,
        )

        decision = interpreter.interpret(event)
        response = executor.respond(decision, event)

        if origin == "cli":
            return ActionResult(
                intention_key="NOTIFY_CLI",
                payload={
                    "message": response,
                },
                urgency="normal",
            )

        if origin == "api":
            return ActionResult(
                intention_key="NOTIFY_API",
                payload={
                    "correlation_id": correlation_id,
                    "message": response,
                    "data": {"decision": decision.__dict__},
                },
                urgency="normal",
            )

        return ActionResult(
            intention_key="NOTIFY_TELEGRAM",
            payload={
                "chat_id": chat_id,
                "message": response,
            },
            urgency="normal",
        )
