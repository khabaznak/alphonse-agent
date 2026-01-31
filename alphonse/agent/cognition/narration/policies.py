from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from alphonse.agent.cognition.narration.models import (
    AudienceRef,
    ContextBundle,
    ModelPlan,
    NarrationIntent,
    PresentationSpec,
)
from alphonse.agent.identity import store as identity_store


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
        return ModelPlan(
            provider="local",
            model="mistral:7b-instruct",
            max_tokens=256,
            temperature=0.4,
            timeout_ms=8000,
            fallback=[],
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
    if audience.kind == "person":
        prefs = identity_store.list_prefs_for_person(audience.id)
        if prefs:
            return prefs
        groups = identity_store.list_person_groups(audience.id)
        for group in groups:
            prefs = identity_store.list_prefs_for_group(group["group_id"])
            if prefs:
                return prefs
    if audience.kind == "group":
        prefs = identity_store.list_prefs_for_group(audience.id)
        if prefs:
            return prefs
    return context.identity.get("defaults", {})


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
