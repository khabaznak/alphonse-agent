from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from alphonse.agent.io.adapters import ExtremityAdapter, SenseAdapter
from alphonse.agent.io.contracts import NormalizedInboundMessage, NormalizedOutboundMessage
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.integrations.domotics import ActionRequest, get_domotics_facade

logger = get_component_logger("io.homeassistant_channel")


@dataclass(frozen=True)
class HomeAssistantSenseAdapter(SenseAdapter):
    channel_type: str = "homeassistant"

    def normalize(self, payload: dict[str, Any]) -> NormalizedInboundMessage:
        entity_id = str(payload.get("entity_id") or "").strip() or None
        new_state = payload.get("new_state")
        text = f"{entity_id or 'unknown'} -> {new_state}" if new_state is not None else (entity_id or "homeassistant")
        return NormalizedInboundMessage(
            text=text,
            channel_type=self.channel_type,
            channel_target=entity_id,
            user_id=None,
            user_name="homeassistant",
            timestamp=time.time(),
            correlation_id=entity_id,
            metadata=payload if isinstance(payload, dict) else {},
        )


class HomeAssistantExtremityAdapter(ExtremityAdapter):
    channel_type: str = "homeassistant"

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        facade = get_domotics_facade()
        if facade is None:
            logger.info("HomeAssistant integration disabled (missing config)")
            return

        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        domain = str(metadata.get("domain") or "").strip()
        service = str(metadata.get("service") or "").strip()
        if not domain or not service:
            logger.warning("HomeAssistantExtremityAdapter skipped: missing domain/service metadata")
            return

        data = metadata.get("data") if isinstance(metadata.get("data"), dict) else {}
        target = metadata.get("target") if isinstance(metadata.get("target"), dict) else {}
        readback = bool(metadata.get("readback", True))
        expected_attributes = (
            metadata.get("expected_attributes")
            if isinstance(metadata.get("expected_attributes"), dict)
            else {}
        )

        result = facade.execute(
            ActionRequest(
                action_type="call_service",
                domain=domain,
                service=service,
                data=data,
                target=target,
                readback=readback,
                readback_entity_id=(str(metadata.get("readback_entity_id")) if metadata.get("readback_entity_id") else None),
                expected_state=(str(metadata.get("expected_state")) if metadata.get("expected_state") is not None else None),
                expected_attributes=expected_attributes,
            )
        )

        if not result.transport_ok:
            logger.warning(
                "HomeAssistantExtremityAdapter transport failure code=%s detail=%s",
                result.error_code,
                result.error_detail,
            )
            return

        logger.info(
            "HomeAssistantExtremityAdapter action delivered domain=%s service=%s effect_applied_ok=%s",
            domain,
            service,
            result.effect_applied_ok,
        )
