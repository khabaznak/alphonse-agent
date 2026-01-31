from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from fastapi import FastAPI, Body

from alphonse.agent.runtime import get_runtime
from alphonse.agent.cognition.skills.interpretation.interpreter import MessageInterpreter
from alphonse.agent.cognition.skills.interpretation.models import MessageEvent, RoutingDecision
from alphonse.agent.cognition.skills.interpretation.registry import build_default_registry
from alphonse.agent.cognition.skills.interpretation.skills import SkillExecutor, build_ollama_client
from alphonse.agent.nervous_system.timed_store import list_timed_signals

app = FastAPI(title="Alphonse API", version="0.1.0")

_REGISTRY = build_default_registry()
_LLM_CLIENT = build_ollama_client()
_INTERPRETER = MessageInterpreter(_REGISTRY, _LLM_CLIENT)
_EXECUTOR = SkillExecutor(_REGISTRY, _LLM_CLIENT)


@app.get("/agent/status")
def agent_status() -> dict[str, object]:
    return get_runtime().snapshot()


@app.post("/agent/message")
def agent_message(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    args = payload.get("args")
    timestamp = payload.get("timestamp")
    event = MessageEvent(
        text=text,
        user_id=_as_optional_str(payload.get("user_id")),
        channel=str(payload.get("channel") or "webui"),
        timestamp=float(timestamp) if timestamp is not None else time.time(),
        metadata={
            **dict(payload.get("metadata") or {}),
            **({"args": args} if isinstance(args, dict) else {}),
        },
    )
    decision = _INTERPRETER.interpret(event)
    response = _EXECUTOR.respond(decision, event)
    return {
        "decision": _decision_dict(decision),
        "response": response,
    }


@app.get("/agent/timed-signals")
def timed_signals(limit: int = 200) -> dict[str, Any]:
    return {"timed_signals": list_timed_signals(limit=limit)}


def _decision_dict(decision: RoutingDecision) -> dict[str, Any]:
    data = asdict(decision)
    data["path"] = decision.path
    return data


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)
