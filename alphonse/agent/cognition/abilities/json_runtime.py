from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alphonse.agent.cognition.abilities.registry import Ability
from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.config import settings

logger = logging.getLogger(__name__)


def load_json_abilities() -> list[Ability]:
    specs = _load_specs_file()
    abilities: list[Ability] = []
    for spec in specs:
        ability = _ability_from_spec(spec)
        if ability is not None:
            abilities.append(ability)
    return abilities


def _load_specs_file() -> list[dict[str, Any]]:
    path = (
        Path(__file__).resolve().parents[2]
        / "nervous_system"
        / "resources"
        / "ability_specs.seed.json"
    )
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("ability spec load failed path=%s error=%s", path, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("ability specs root must be list path=%s", path)
        return []
    return [item for item in raw if isinstance(item, dict)]


def _ability_from_spec(spec: dict[str, Any]) -> Ability | None:
    kind = str(spec.get("kind") or "").strip()
    intent_name = str(spec.get("intent_name") or "").strip()
    tools_raw = spec.get("tools")
    tools = tuple(str(item).strip() for item in tools_raw) if isinstance(tools_raw, list) else tuple()
    if not intent_name or not kind:
        return None
    if kind == "clock_time_response":
        return Ability(intent_name=intent_name, tools=tools, execute=_build_clock_time_executor(spec))
    if kind == "plan_emit":
        return Ability(intent_name=intent_name, tools=tools, execute=_build_plan_emit_executor(spec))
    logger.warning("unsupported ability spec kind=%s intent=%s", kind, intent_name)
    return None


def _build_clock_time_executor(spec: dict[str, Any]):
    timezone_key = str(spec.get("timezone_state_key") or "timezone")
    timezone_default = str(spec.get("timezone_default") or settings.get_timezone())
    responses = spec.get("responses") if isinstance(spec.get("responses"), dict) else {}

    def _execute(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
        clock = tools.get("clock")
        if clock is None:
            return {"response_key": "generic.unknown"}
        timezone_name = str(state.get(timezone_key) or timezone_default)
        now = clock.current_time(timezone_name)
        locale = str(state.get("locale") or "")
        language = "es" if locale.startswith("es") else "default"
        template = str(responses.get(language) or responses.get("default") or "It is {time} in {timezone}.")
        return {
            "response_text": template.format(
                time=now.strftime("%H:%M"),
                timezone=timezone_name,
            )
        }

    return _execute


def _build_plan_emit_executor(spec: dict[str, Any]):
    plan_spec = spec.get("plan") if isinstance(spec.get("plan"), dict) else {}
    plan_type_raw = str(plan_spec.get("plan_type") or "").strip().upper()
    payload_spec = plan_spec.get("payload") if isinstance(plan_spec.get("payload"), dict) else {}
    target_from = plan_spec.get("target_from") if isinstance(plan_spec.get("target_from"), list) else []
    channels_from = plan_spec.get("channels_from") if isinstance(plan_spec.get("channels_from"), list) else []
    fallback_target = plan_spec.get("target")
    priority = int(plan_spec.get("priority") or 0)

    def _execute(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
        _ = tools
        if not plan_type_raw:
            return {"response_key": "generic.unknown"}
        try:
            plan_type = PlanType(plan_type_raw)
        except ValueError:
            return {"response_key": "generic.unknown"}

        target = _resolve_target(state, target_from, fallback_target)
        channels = _resolve_channels(state, channels_from)
        payload = _resolve_value(payload_spec, state)
        if not isinstance(payload, dict):
            payload = {}
        plan = CortexPlan(
            plan_type=plan_type,
            priority=priority,
            target=target,
            channels=channels,
            payload=payload,
        )
        return {"plans": [plan.model_dump()]}

    return _execute


def _resolve_target(
    state: dict[str, Any],
    source_keys: list[Any],
    fallback_target: Any,
) -> str | None:
    for raw in source_keys:
        key = str(raw or "").strip()
        if not key:
            continue
        value = state.get(key)
        if value is None:
            continue
        return str(value)
    if fallback_target is None:
        return None
    return str(_resolve_value(fallback_target, state))


def _resolve_channels(state: dict[str, Any], source_keys: list[Any]) -> list[str] | None:
    channels: list[str] = []
    for raw in source_keys:
        key = str(raw or "").strip()
        if not key:
            continue
        value = state.get(key)
        if value is None:
            continue
        channels.append(str(value))
    return channels or None


def _resolve_value(value: Any, state: dict[str, Any]) -> Any:
    if isinstance(value, str):
        token = _extract_state_token(value)
        if token:
            return state.get(token)
        return value
    if isinstance(value, list):
        return [_resolve_value(item, state) for item in value]
    if isinstance(value, dict):
        return {str(k): _resolve_value(v, state) for k, v in value.items()}
    return value


def _extract_state_token(value: str) -> str | None:
    text = value.strip()
    if text.startswith("{") and text.endswith("}") and len(text) > 2:
        return text[1:-1].strip()
    return None
