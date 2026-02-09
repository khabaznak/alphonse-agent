from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphonse.agent.cognition.abilities.registry import Ability
from alphonse.agent.cognition.abilities.store import AbilitySpecStore
from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.cognition.pending_interaction import (
    PendingInteraction,
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
    try_consume,
)
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_or_create_scope_principal,
    get_preference,
    set_preference,
)
from alphonse.agent.nervous_system.onboarding_profiles import upsert_onboarding_profile
from alphonse.agent.nervous_system.location_profiles import (
    insert_device_location,
    list_device_locations,
    list_location_profiles,
    upsert_location_profile,
)
from alphonse.agent.nervous_system.users import get_user_by_display_name, list_users, upsert_user
from alphonse.agent.nervous_system.telegram_invites import update_invite_status
from alphonse.agent.identity.store import upsert_person, upsert_channel
from alphonse.agent.identity import profile as identity_profile
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.config import settings

logger = logging.getLogger(__name__)


def load_json_abilities(db_path: str | None = None) -> list[Ability]:
    specs = _merge_specs(
        _load_specs_file(),
        _load_specs_db(db_path),
    )
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


def _load_specs_db(db_path: str | None = None) -> list[dict[str, Any]]:
    try:
        store = AbilitySpecStore(db_path=db_path)
        return store.list_enabled_specs()
    except Exception as exc:
        logger.warning("ability specs db load failed db_path=%s error=%s", db_path, exc)
        return []


def _merge_specs(
    file_specs: list[dict[str, Any]],
    db_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for spec in file_specs:
        key = str(spec.get("intent_name") or "").strip()
        if not key:
            continue
        merged[key] = spec
    for spec in db_specs:
        key = str(spec.get("intent_name") or "").strip()
        if not key:
            continue
        merged[key] = spec
    return [merged[key] for key in sorted(merged.keys())]


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
    if kind == "tool_call_then_response":
        return Ability(intent_name=intent_name, tools=tools, execute=_build_tool_call_then_response_executor(spec))
    if kind == "ability_flow":
        return Ability(intent_name=intent_name, tools=tools, execute=_build_ability_flow_executor(spec))
    logger.warning("unsupported ability spec kind=%s intent=%s", kind, intent_name)
    return None


def _build_ability_flow_executor(spec: dict[str, Any]):
    intent_name = str(spec.get("intent_name") or "")
    parameters = spec.get("input_parameters") if isinstance(spec.get("input_parameters"), list) else []
    steps = spec.get("step_sequence") if isinstance(spec.get("step_sequence"), list) else []
    prompts = spec.get("prompts") if isinstance(spec.get("prompts"), dict) else {}
    outputs = spec.get("outputs") if isinstance(spec.get("outputs"), dict) else {}

    def _execute(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
        _ = tools
        ability_state = _load_ability_state(state, intent_name)
        params = ability_state["params"]
        last_text = str(state.get("last_user_message") or "").strip()
        pending = _parse_pending_interaction(state.get("pending_interaction"))
        _prefill_params(params, parameters, state)
        if pending and pending.context.get("ability_intent") == intent_name:
            resolution = try_consume(last_text, pending)
            if resolution.consumed:
                _merge_pending_result(params, pending.key, resolution.result or {})
                ability_state["pending_param"] = None
                state["pending_interaction"] = None
        _extract_params_from_text(params, parameters, last_text, state)
        missing = _missing_required_params(params, parameters)
        if missing:
            next_param = missing[0]
            pending = build_pending_interaction(
                PendingInteractionType.SLOT_FILL,
                key=next_param,
                context={"ability_intent": intent_name, "param": next_param},
            )
            ability_state["pending_param"] = next_param
            return {
                "response_key": str(prompts.get(next_param) or "clarify.intent"),
                "pending_interaction": serialize_pending_interaction(pending),
                "ability_state": ability_state,
            }
        step_result = _execute_steps(steps, params, state)
        if isinstance(step_result, dict) and step_result.get("response_key"):
            pending_interaction = step_result.get("pending_interaction")
            return {
                "response_key": step_result.get("response_key"),
                "response_vars": step_result.get("response_vars") or {},
                "pending_interaction": pending_interaction,
                "ability_state": ability_state if pending_interaction else {},
            }
        response_key = str(outputs.get("success") or "ack.confirmed")
        response_vars = dict(step_result.get("response_vars") or {})
        if "user_name" not in response_vars:
            if isinstance(params.get("user_name"), str):
                response_vars["user_name"] = params.get("user_name")
            elif isinstance(params.get("display_name"), str):
                response_vars["user_name"] = params.get("display_name")
        return {
            "response_key": response_key,
            "response_vars": response_vars,
            "pending_interaction": None,
            "ability_state": {},
        }

    return _execute


def _load_ability_state(state: dict[str, Any], intent_name: str) -> dict[str, Any]:
    ability_state = state.get("ability_state")
    if not isinstance(ability_state, dict):
        ability_state = {}
    if ability_state.get("intent") != intent_name:
        return {"intent": intent_name, "params": {}, "pending_param": None}
    params = ability_state.get("params")
    if not isinstance(params, dict):
        ability_state["params"] = {}
    return ability_state


def _parse_pending_interaction(raw: dict[str, Any] | None) -> PendingInteraction | None:
    if not isinstance(raw, dict):
        return None
    try:
        raw_type = str(raw.get("type") or "")
        pending_type = PendingInteractionType(raw_type.split(".")[-1].upper())
        return PendingInteraction(
            type=pending_type,
            key=str(raw.get("key") or ""),
            context=raw.get("context") or {},
            created_at=str(raw.get("created_at") or ""),
            expires_at=raw.get("expires_at"),
        )
    except Exception:
        return None


def _merge_pending_result(params: dict[str, Any], key: str, result: dict[str, Any]) -> None:
    if key in result:
        params[key] = result.get(key)
        if key == "label":
            normalized = _normalize_location_label(params.get("label"))
            if normalized:
                params["label"] = normalized
        return
    if result:
        params[key] = list(result.values())[0]
        if key == "label":
            normalized = _normalize_location_label(params.get("label"))
            if normalized:
                params["label"] = normalized


def _missing_required_params(params: dict[str, Any], parameters: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for param in sorted(parameters, key=lambda p: int(p.get("order") or 0)):
        name = str(param.get("name") or "").strip()
        if not name:
            continue
        required = bool(param.get("required", False))
        if required and name not in params:
            missing.append(name)
    return missing


def _extract_params_from_text(
    params: dict[str, Any],
    parameters: list[dict[str, Any]],
    text: str,
    state: dict[str, Any],
) -> None:
    if not text:
        return
    missing_required = _missing_required_params(params, parameters)
    relationship = _extract_relationship(text)
    if relationship and "relationship" in missing_required:
        params.setdefault("relationship", relationship)
    name = _extract_name_after_relationship(text)
    if name and "display_name" in missing_required:
        params.setdefault("display_name", name)
    if len(missing_required) == 1:
        key = missing_required[0]
        param_type = _param_type(parameters, key)
        if param_type == "person_name" and _looks_like_name(text):
            params.setdefault(key, text)
    if "channel_provider" in missing_required:
        channel_provider = _extract_channel_type(text)
        if channel_provider:
            params.setdefault("channel_provider", channel_provider)
    if "channel_address" in missing_required:
        channel_address = _extract_channel_address(text)
        if channel_address:
            params.setdefault("channel_address", channel_address)
    if "channel_address" in missing_required:
        reply_id = state.get("incoming_reply_to_user_id")
        if reply_id:
            params.setdefault("channel_address", str(reply_id))
    if "channel_address" in missing_required:
        chat_id = state.get("channel_target") or state.get("chat_id")
        if chat_id:
            params.setdefault("channel_address", str(chat_id))
    if "label" not in params and _param_type(parameters, "label"):
        label = _extract_location_label(text)
        if label:
            params.setdefault("label", label)
    if "address_text" not in params and _param_type(parameters, "address_text"):
        address = _extract_address_text(text)
        if address:
            params.setdefault("address_text", address)


def _param_type(parameters: list[dict[str, Any]], key: str) -> str:
    for param in parameters:
        if str(param.get("name") or "") == key:
            return str(param.get("type") or "")
    return ""


def _looks_like_name(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 40:
        return False
    parts = [p for p in stripped.split() if p]
    if len(parts) > 3:
        return False
    for part in parts:
        if not part[0].isalpha():
            return False
    return True


def _extract_relationship(text: str) -> str | None:
    normalized = text.lower()
    mapping = {
        "wife": "wife",
        "husband": "husband",
        "son": "son",
        "daughter": "daughter",
        "mom": "mother",
        "mother": "mother",
        "dad": "father",
        "father": "father",
        "esposa": "wife",
        "esposo": "husband",
        "hijo": "son",
        "hija": "daughter",
        "madre": "mother",
        "padre": "father",
    }
    for key, value in mapping.items():
        if key in normalized:
            return value
    return None


def _extract_name_after_relationship(text: str) -> str | None:
    tokens = [t for t in text.replace(",", " ").split() if t]
    if not tokens:
        return None
    for idx, token in enumerate(tokens):
        lower = token.lower()
        if lower in {
            "wife",
            "husband",
            "son",
            "daughter",
            "mom",
            "mother",
            "dad",
            "father",
            "esposa",
            "esposo",
            "hijo",
            "hija",
            "madre",
            "padre",
        }:
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
    return None


def _extract_channel_type(text: str) -> str | None:
    normalized = text.lower()
    if "telegram" in normalized:
        return "telegram"
    if "discord" in normalized:
        return "discord"
    if "whatsapp" in normalized:
        return "whatsapp"
    if "sms" in normalized:
        return "sms"
    return None


def _extract_channel_address(text: str) -> str | None:
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch == " ")
    digits = "".join(cleaned.split())
    if len(digits) >= 5:
        return digits
    return None


def _extract_location_label(text: str) -> str | None:
    normalized = text.lower()
    mapping = {
        "home": "home",
        "house": "home",
        "casa": "home",
        "work": "work",
        "office": "work",
        "trabajo": "work",
        "other": "other",
        "otra": "other",
    }
    for key, value in mapping.items():
        if key in normalized:
            return value
    return None


def _normalize_location_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"home", "work", "other"}:
        return normalized
    return _extract_location_label(normalized)


def _extract_address_text(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) < 6:
        return None
    return stripped


def _execute_steps(steps: list[dict[str, Any]], params: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    for step in sorted(steps, key=lambda s: int(s.get("order") or 0)):
        action = str(step.get("action") or "").strip()
        if not action:
            continue
        if action == "identity.set_display_name":
            name = str(params.get("user_name") or params.get("display_name") or "")
            if name:
                identity_profile.set_display_name(str(state.get("conversation_key") or ""), name)
            continue
        if action == "users.upsert_admin":
            name = str(params.get("user_name") or params.get("display_name") or "")
            principal_id = _principal_id_from_state(state)
            if principal_id and name:
                upsert_user(
                    {
                        "user_id": principal_id,
                        "principal_id": principal_id,
                        "display_name": name,
                        "is_admin": True,
                        "is_active": True,
                        "onboarded_at": _now(),
                    }
                )
            continue
        if action == "users.upsert_secondary":
            name = str(params.get("display_name") or "")
            if name:
                user_id = str(params.get("user_id") or "")
                if not user_id:
                    user_id = str(uuid.uuid4())
                    params["user_id"] = user_id
                upsert_user(
                    {
                        "user_id": user_id,
                        "principal_id": None,
                        "display_name": name,
                        "role": str(params.get("role") or ""),
                        "relationship": str(params.get("relationship") or ""),
                        "is_admin": False,
                        "is_active": True,
                        "onboarded_at": _now(),
                    }
                )
            continue
        if action == "onboarding.mark_primary_complete":
            principal_id = _principal_id_from_state(state)
            system_principal_id = get_or_create_scope_principal("system", "default")
            if principal_id and system_principal_id:
                set_preference(system_principal_id, "onboarding.primary.completed", True, source="system")
                set_preference(system_principal_id, "onboarding.primary.admin_principal_id", principal_id, source="system")
                set_preference(principal_id, "onboarding.state", "completed", source="system")
                upsert_onboarding_profile(
                    {
                        "principal_id": principal_id,
                        "state": "in_progress",
                        "primary_role": "admin",
                        "next_steps": [
                            "role",
                            "relationship",
                            "home_location",
                            "work_location",
                        ],
                    }
                )
            continue
        if action == "channels.authorize":
            display_name = str(params.get("display_name") or params.get("user_name") or "").strip()
            if not display_name:
                return {"response_key": "core.onboarding.authorize.user_not_found"}
            principal_id = _principal_id_from_state(state)
            system_principal_id = get_or_create_scope_principal("system", "default")
            if system_principal_id and principal_id:
                admin_id = get_preference(system_principal_id, "onboarding.primary.admin_principal_id")
                if admin_id and str(admin_id) != str(principal_id):
                    return {"response_key": "policy.authorize.restricted"}
            user = get_user_by_display_name(display_name)
            if not user:
                return {
                    "response_key": "core.onboarding.authorize.user_not_found",
                    "response_vars": {"user_name": display_name},
                }
            channel_provider = str(params.get("channel_provider") or "telegram").strip()
            channel_address = str(params.get("channel_address") or "").strip()
            if not channel_address:
                channel_address = str(state.get("incoming_reply_to_user_id") or "").strip()
            if not channel_address:
                return {"response_key": "core.onboarding.authorize.ask_channel_address"}
            if not display_name:
                display_name = str(state.get("incoming_reply_to_user_name") or "").strip()
            person_id = str(user.get("user_id") or "")
            if person_id:
                upsert_person(
                    {
                        "person_id": person_id,
                        "display_name": user.get("display_name") or display_name,
                        "relationship": user.get("relationship"),
                        "is_active": user.get("is_active", True),
                    }
                )
                channel_id = f"{channel_provider}:{channel_address}"
                upsert_channel(
                    {
                        "channel_id": channel_id,
                        "channel_type": channel_provider,
                        "person_id": person_id,
                        "address": channel_address,
                        "is_enabled": True,
                        "priority": 100,
                    }
                )
            return {
                "response_vars": {
                    "user_name": display_name,
                    "channel_provider": channel_provider,
                }
            }
        if action == "telegram.invite_approve":
            channel_address = str(params.get("channel_address") or "").strip()
            if channel_address:
                update_invite_status(channel_address, "approved")
            return {}
        if action == "users.list":
            rows = list_users(active_only=True, limit=50)
            if not rows:
                return {"response_text": "No users yet."}
            lines = []
            for row in rows:
                name = row.get("display_name") or "Unknown"
                role = row.get("role") or ""
                relationship = row.get("relationship") or ""
                admin = "admin" if row.get("is_admin") else ""
                bits = [name]
                if role:
                    bits.append(role)
                if relationship:
                    bits.append(relationship)
                if admin:
                    bits.append(admin)
                lines.append(" - ".join(bits))
            return {"response_key": "core.users.list", "response_vars": {"lines": lines}}
        if action == "location.current":
            principal_id = _principal_id_from_state(state)
            label = str(params.get("label") or "").strip().lower() or None
            if label:
                label = _normalize_location_label(label) or label
            location_text = None
            if principal_id:
                device = list_device_locations(principal_id=principal_id, limit=1)
                if device:
                    entry = device[0]
                    location_text = f"{entry.get('latitude')}, {entry.get('longitude')}"
            if not location_text and principal_id and label:
                profiles = list_location_profiles(principal_id=principal_id, label=label, active_only=True, limit=1)
                if profiles:
                    profile = profiles[0]
                    location_text = profile.get("address_text") or f"{profile.get('latitude')}, {profile.get('longitude')}"
            if not location_text and label:
                return {
                    "response_key": "core.location.current.not_set",
                    "response_vars": {"label": label},
                }
            if not location_text:
                pending = build_pending_interaction(
                    PendingInteractionType.SLOT_FILL,
                    key="label",
                    context={"ability_intent": "core.location.current", "param": "label"},
                )
                return {
                    "response_key": "core.location.current.ask_label",
                    "pending_interaction": serialize_pending_interaction(pending),
                }
            return {
                "response_key": "core.location.current",
                "response_vars": {"location": location_text},
            }
        if action == "location.set":
            principal_id = _principal_id_from_state(state)
            if not principal_id:
                return {"response_key": "generic.unknown"}
            label = str(params.get("label") or "").strip().lower()
            if label not in {"home", "work", "other"}:
                return {"response_key": "core.location.set.ask_label"}
            address_text = str(params.get("address_text") or "").strip()
            if not address_text:
                return {"response_key": "core.location.set.ask_address"}
            upsert_location_profile(
                {
                    "principal_id": principal_id,
                    "label": label,
                    "address_text": address_text,
                    "source": "user",
                    "confidence": 0.9,
                }
            )
            return {
                "response_key": "core.location.set.completed",
                "response_vars": {"label": label},
            }
    return {}


def _prefill_params(params: dict[str, Any], parameters: list[dict[str, Any]], state: dict[str, Any]) -> None:
    for param in parameters:
        name = str(param.get("name") or "")
        if not name or name in params:
            continue
        if "default" in param:
            params[name] = param.get("default")
            continue
        if name == "user_name":
            conversation_key = str(state.get("conversation_key") or "")
            if conversation_key:
                existing = identity_profile.get_display_name(conversation_key)
                if existing:
                    params[name] = existing
        if name == "channel_provider":
            channel_hint = str(state.get("channel_type") or "").strip().lower()
            if channel_hint:
                params[name] = channel_hint
            else:
                params[name] = "telegram"
        if name == "channel_address":
            reply_id = state.get("incoming_reply_to_user_id")
            if reply_id:
                params[name] = str(reply_id)
        if name == "display_name":
            reply_name = state.get("incoming_reply_to_user_name")
            if reply_name:
                params[name] = str(reply_name)


def _principal_id_from_state(state: dict[str, Any]) -> str | None:
    channel_type = str(state.get("channel_type") or "")
    channel_target = str(state.get("channel_target") or state.get("chat_id") or "")
    if not channel_type or not channel_target:
        return None
    return get_or_create_principal_for_channel(channel_type, channel_target)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _build_tool_call_then_response_executor(spec: dict[str, Any]):
    tool_call = spec.get("tool_call") if isinstance(spec.get("tool_call"), dict) else {}
    responses = spec.get("responses") if isinstance(spec.get("responses"), dict) else {}
    response_key = spec.get("response_key")

    def _execute(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
        output = _execute_tool_call(tool_call, state, tools)
        if output is None:
            if isinstance(response_key, str) and response_key.strip():
                return {"response_key": response_key.strip()}
            return {"response_key": "generic.unknown"}
        locale = str(state.get("locale") or "")
        language = "es" if locale.startswith("es") else "default"
        template = str(
            responses.get(language) or responses.get("default") or "{output}"
        )
        response_text = _render_template(
            template,
            {
                "state": state,
                "tool_output": output,
                "locale": locale,
            },
        )
        return {"response_text": response_text}

    return _execute


def _execute_tool_call(
    tool_call: dict[str, Any], state: dict[str, Any], tools: ToolRegistry
) -> dict[str, Any] | None:
    tool_name = str(tool_call.get("tool") or "").strip()
    method_name = str(tool_call.get("method") or "").strip()
    args_spec = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
    if not tool_name or not method_name:
        return None
    tool = tools.get(tool_name)
    if tool is None:
        return None

    # Constrained dispatch: only explicitly allowed tool-method pairs.
    if tool_name == "clock" and method_name == "current_time":
        timezone_name = str(
            _resolve_value(args_spec.get("timezone_name"), state) or settings.get_timezone()
        )
        result = tool.current_time(timezone_name)
        return {
            "time": result.strftime("%H:%M"),
            "timezone": timezone_name,
            "iso": result.isoformat(),
        }
    return None


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
        token = _extract_token(value)
        if token:
            return _resolve_token(token, state)
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


def _extract_token(value: str) -> str | None:
    return _extract_state_token(value)


def _resolve_token(token: str, state: dict[str, Any]) -> Any:
    if not token:
        return None
    if "." not in token:
        return state.get(token)
    parts = token.split(".")
    current: Any = state
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _render_template(template: str, context: dict[str, Any]) -> str:
    def _replace(match: Any) -> str:
        token = str(match.group(1) or "").strip()
        if not token:
            return ""
        if token in context:
            value = context.get(token)
            return "" if value is None else str(value)
        value = _resolve_from_context(token, context)
        return "" if value is None else str(value)

    import re

    return re.sub(r"\{([^{}]+)\}", _replace, template)


def _resolve_from_context(path: str, context: dict[str, Any]) -> Any:
    parts = path.split(".")
    if not parts:
        return None
    current: Any = context
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
