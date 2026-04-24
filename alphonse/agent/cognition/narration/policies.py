from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from alphonse.agent import identity
from alphonse.agent.cognition.narration.models import (
    AudienceRef,
    ContextBundle,
    ModelPlan,
    NarrationIntent,
    PresentationSpec,
)
from alphonse.agent.cognition.preferences.store import get_user_preference


@dataclass
class PolicyStack:
    behavior: "NarrationBehaviorPolicy"
    preferences: "CommunicationPreferencesPolicy"
    model_routing: "ModelRoutingPolicy"

    def evaluate(self, context: ContextBundle) -> tuple[NarrationIntent, PresentationSpec, ModelPlan]:
        intent = self.behavior.evaluate(context)
        presentation = self.preferences.evaluate(context, intent)
        model_plan = self.model_routing.evaluate(context, intent, presentation)
        return intent, presentation, model_plan


class NarrationBehaviorPolicy:
    def __init__(self) -> None:
        self._last_sent: dict[str, float] = {}

    def evaluate(self, context: ContextBundle) -> NarrationIntent:
        event = context.event
        origin = str(event.get("origin") or "system")
        audience = _resolve_audience(context)

        channel_hint = context.identity.get("channel_hint")
        channel_type = str(channel_hint or _channel_from_origin(origin))
        should_narrate = True

        prefs = _resolve_prefs(context, audience)
        quiet = _in_quiet_hours(prefs, context)
        in_meeting = bool(context.presence.get("in_meeting"))
        if quiet or in_meeting:
            channel_type = "silent"
            should_narrate = False
        if not _allow_channel(prefs, channel_type):
            channel_type = "silent"
            should_narrate = False

        if should_narrate and not _allow_rate(context, self._last_sent, audience, channel_type):
            channel_type = "silent"
            should_narrate = False

        return NarrationIntent(
            should_narrate=should_narrate,
            audience=audience,
            channel_type=channel_type,
            priority=_priority_from_event(event),
            timing="now",
            verbosity="normal",
            format="plain",
            reason="behavior policy",
        )


class CommunicationPreferencesPolicy:
    def evaluate(self, context: ContextBundle, intent: NarrationIntent) -> PresentationSpec:
        prefs = _resolve_prefs(context, intent.audience)
        return PresentationSpec(
            language=str(prefs.get("language_preference") or "en"),
            tone=str(prefs.get("tone") or "neutral"),
            formality=str(prefs.get("formality") or "neutral"),
            emoji=str(prefs.get("emoji") or "none"),
            verbosity_cap=str(prefs.get("verbosity_cap") or "normal"),
            safety_mode="strict",
            reason="preferences policy",
        )


class ModelRoutingPolicy:
    def evaluate(
        self,
        _context: ContextBundle,
        _intent: NarrationIntent,
        presentation: PresentationSpec,
    ) -> ModelPlan:
        policy = presentation.verbosity_cap
        budget = str(_context.identity.get("model_budget_policy") or "local_only")
        provider = "local"
        model = "mistral:7b-instruct"
        if budget == "openai_allowed" and _intent.priority in {"high", "urgent"}:
            provider = "openai"
            model = "gpt-4o-mini"
        fallback = [{"provider": "local", "model": "mistral:7b-instruct"}]
        return ModelPlan(
            provider=provider,
            model=model,
            max_tokens=256,
            temperature=0.4,
            timeout_ms=8000,
            fallback=fallback,
            reason=f"model policy {policy}",
        )


def _resolve_audience(context: ContextBundle) -> AudienceRef:
    identity = context.identity
    if identity.get("person_id"):
        return AudienceRef(kind="person", id=str(identity["person_id"]))
    if identity.get("group_id"):
        return AudienceRef(kind="group", id=str(identity["group_id"]))
    return AudienceRef(kind="system", id="system")


def _resolve_prefs(context: ContextBundle, audience: AudienceRef) -> dict[str, Any]:
    defaults = dict(context.identity.get("defaults", {}))
    if audience.kind != "person":
        return defaults
    user = identity.get_user(audience.id)
    user_id = str((user or {}).get("user_id") or "").strip()
    if not user_id:
        return defaults
    return {
        "language_preference": get_user_preference(user_id, "language_preference") or defaults.get("language_preference", "en"),
        "tone": get_user_preference(user_id, "tone") or defaults.get("tone", "neutral"),
        "formality": get_user_preference(user_id, "formality") or defaults.get("formality", "neutral"),
        "emoji": get_user_preference(user_id, "emoji") or defaults.get("emoji", "none"),
        "verbosity_cap": get_user_preference(user_id, "verbosity_cap") or defaults.get("verbosity_cap", "normal"),
        "quiet_hours_start": get_user_preference(user_id, "quiet_hours_start") if get_user_preference(user_id, "quiet_hours_start") is not None else defaults.get("quiet_hours_start"),
        "quiet_hours_end": get_user_preference(user_id, "quiet_hours_end") if get_user_preference(user_id, "quiet_hours_end") is not None else defaults.get("quiet_hours_end"),
        "allow_telegram": get_user_preference(user_id, "allow_telegram") if get_user_preference(user_id, "allow_telegram") is not None else True,
        "allow_web": get_user_preference(user_id, "allow_web") if get_user_preference(user_id, "allow_web") is not None else defaults.get("allow_web", True),
        "allow_cli": get_user_preference(user_id, "allow_cli") if get_user_preference(user_id, "allow_cli") is not None else defaults.get("allow_cli", True),
        "allow_push": get_user_preference(user_id, "allow_push") if get_user_preference(user_id, "allow_push") is not None else defaults.get("allow_push", True),
    }


def _in_quiet_hours(prefs: dict[str, Any], context: ContextBundle) -> bool:
    start = prefs.get("quiet_hours_start")
    end = prefs.get("quiet_hours_end")
    if start is None or end is None:
        return False
    hour = int(context.time_context.get("local_hour", 0))
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _priority_from_event(event: dict[str, Any]) -> str:
    return str(event.get("severity") or "normal")


def _channel_from_origin(origin: str) -> str:
    if origin in {"telegram", "cli", "api", "web"}:
        return origin
    return "silent"


def _allow_channel(prefs: dict[str, Any], channel_type: str) -> bool:
    if channel_type == "telegram":
        return bool(prefs.get("allow_telegram", True))
    if channel_type == "web":
        return bool(prefs.get("allow_web", True))
    if channel_type == "cli":
        return bool(prefs.get("allow_cli", True))
    if channel_type == "push":
        return bool(prefs.get("allow_push", True))
    return True


def _allow_rate(
    context: ContextBundle,
    last_sent: dict[str, float],
    audience: AudienceRef,
    channel_type: str,
) -> bool:
    key = f"{audience.kind}:{audience.id}:{channel_type}"
    now_ts = _timestamp(context)
    min_interval = 30.0
    last = last_sent.get(key)
    if last and now_ts - last < min_interval:
        return False
    last_sent[key] = now_ts
    return True


def _timestamp(context: ContextBundle) -> float:
    now_iso = context.time_context.get("now_iso")
    try:
        return datetime.fromisoformat(str(now_iso)).timestamp()
    except Exception:
        return datetime.now().timestamp()
