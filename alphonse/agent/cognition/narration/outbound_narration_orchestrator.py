from __future__ import annotations

from dataclasses import dataclass
import logging
from datetime import datetime
from typing import Any

from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.narration.channel_resolver import resolve_channel
from alphonse.agent.cognition.narration.models import ContextBundle, MessageDraft
from alphonse.agent.cognition.narration.policies import (
    CommunicationPreferencesPolicy,
    ModelRoutingPolicy,
    NarrationBehaviorPolicy,
    PolicyStack,
)
from alphonse.agent.cognition.skills.narration.renderer import render_message
from alphonse.agent.cognition.skills.narration.skill import NarrationSkill
from alphonse.agent.identity import store as identity_store
from alphonse.agent.io import NormalizedOutboundMessage

logger = logging.getLogger(__name__)


@dataclass
class DeliveryCoordinator:
    stack: PolicyStack
    skill: NarrationSkill

    def deliver(self, result: ActionResult, context: dict) -> NormalizedOutboundMessage | None:
        payload = result.payload
        message = payload.get("message")
        if not message:
            return None

        direct_reply = payload.get("direct_reply")
        if isinstance(direct_reply, dict):
            channel_type = str(direct_reply.get("channel_type") or "")
            target = direct_reply.get("target")
            text = direct_reply.get("text")
            correlation_id = direct_reply.get("correlation_id")
            if not target:
                logger.error("DirectReply missing target for channel=%s", channel_type or "unknown")
                raise ValueError("DirectReply requires target")
            if not text:
                logger.error("DirectReply missing text for channel=%s", channel_type or "unknown")
                raise ValueError("DirectReply requires text")
            logger.info(
                "direct reply send attempt channel=%s target=%s text_len=%s correlation_id=%s",
                channel_type or "unknown",
                target,
                len(str(text)),
                correlation_id,
            )
            return _normalized_outbound_for_channel(
                channel_type or "webui",
                str(target),
                str(correlation_id or ""),
                {"content": text},
                audience=_audience_from_payload(payload),
            )

        bundle = build_context_bundle(payload, context)
        intent, presentation, _model_plan = self.stack.evaluate(bundle)
        if not intent.should_narrate or intent.channel_type == "silent":
            return None

        draft = self.skill.compose(
            message=str(message),
            intent=intent,
            presentation=presentation,
            correlation_id=str(payload.get("correlation_id") or bundle.event.get("correlation_id") or ""),
            metadata={"data": payload.get("data")},
        )
        rendered = render_message(draft, presentation)
        channel_hint = payload.get("channel_hint")
        target = payload.get("target")
        if channel_hint and target:
            return _normalized_outbound_for_channel(
                str(channel_hint),
                str(target),
                draft.correlation_id,
                rendered.payload,
                audience=_audience_from_payload(payload),
            )

        resolution = resolve_channel(intent)

        return _normalized_outbound_for_channel(
            rendered.channel_type,
            resolution.address,
            draft.correlation_id,
            rendered.payload,
            audience=_audience_from_payload(payload),
        )


def build_context_bundle(payload: dict[str, Any], context: dict) -> ContextBundle:
    signal = context.get("signal")
    origin = payload.get("origin") or (getattr(signal, "source", None) if signal else None)
    event = {
        "signal_type": getattr(signal, "type", None),
        "payload": getattr(signal, "payload", None),
        "origin": origin or "system",
        "severity": "normal",
        "correlation_id": getattr(signal, "correlation_id", None) if signal else None,
    }
    now = datetime.now().astimezone()
    trace = {"transition_depth_for_correlation_id": payload.get("depth")}
    presence = _resolve_presence(payload)
    identity = _resolve_identity(payload)
    identity["channel_hint"] = payload.get("channel_hint")
    identity["model_budget_policy"] = payload.get("model_budget_policy")
    return ContextBundle(
        event=event,
        trace=trace,
        presence=presence,
        time_context={
            "now_iso": now.isoformat(),
            "local_hour": now.hour,
            "is_night": now.hour < 6 or now.hour >= 22,
        },
        channel_availability={},
        identity=identity,
    )


def _resolve_identity(payload: dict[str, Any]) -> dict[str, Any]:
    audience = payload.get("audience") or {}
    if isinstance(audience, dict) and audience.get("kind") == "person":
        person = identity_store.get_person(str(audience.get("id")))
        if person:
            return {"person_id": person.get("person_id"), "defaults": {}}
    channel_hint = payload.get("channel_hint")
    target = payload.get("target")
    if channel_hint and target:
        person = identity_store.resolve_person_by_channel(str(channel_hint), str(target))
        if person:
            return {"person_id": person.get("person_id"), "defaults": {}}
    return {"defaults": {}}


def _resolve_presence(payload: dict[str, Any]) -> dict[str, Any]:
    audience = payload.get("audience") or {}
    if isinstance(audience, dict) and audience.get("kind") == "person":
        presence = identity_store.get_presence(str(audience.get("id")))
        if presence:
            return presence
    return {"in_meeting": False}


def _normalized_outbound_for_channel(
    channel_type: str,
    address: str | None,
    correlation_id: str,
    payload: dict[str, Any],
    *,
    audience: dict[str, str],
) -> NormalizedOutboundMessage | None:
    normalized_channel = _normalize_channel_type(channel_type)
    target = address or _default_target(normalized_channel)
    if not target:
        return None
    return NormalizedOutboundMessage(
        message=str(payload.get("content") or ""),
        channel_type=normalized_channel,
        channel_target=target,
        audience=audience,
        correlation_id=correlation_id,
        metadata={"data": payload.get("data")},
    )


def _audience_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    audience = payload.get("audience")
    if isinstance(audience, dict):
        kind = audience.get("kind")
        ident = audience.get("id")
        if isinstance(kind, str) and isinstance(ident, str):
            return {"kind": kind, "id": ident}
    return {"kind": "system", "id": "system"}


def _normalize_channel_type(channel_type: str) -> str:
    normalized = str(channel_type or "").strip().lower()
    aliases = {
        "api": "webui",
        "web": "webui",
        "mouth": "voice",
        "local_audio": "voice",
        "audio": "voice",
        "tts": "voice",
    }
    return aliases.get(normalized, normalized or "webui")


def _default_target(channel_type: str) -> str | None:
    defaults = {
        "cli": "cli",
        "webui": "webui",
        "voice": "local",
    }
    return defaults.get(channel_type)


def build_default_coordinator() -> DeliveryCoordinator:
    stack = PolicyStack(
        behavior=NarrationBehaviorPolicy(),
        preferences=CommunicationPreferencesPolicy(),
        model_routing=ModelRoutingPolicy(),
    )
    return DeliveryCoordinator(stack=stack, skill=NarrationSkill())
