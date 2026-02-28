from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.integrations.domotics.contracts import (
    ActionRequest,
    ActionResult,
    NormalizedEvent,
    QueryResult,
    QuerySpec,
    SubscribeSpec,
    SubscriptionHandle,
)
from alphonse.integrations.homeassistant.anti_spam import EventDebouncer, EventFilter
from alphonse.integrations.homeassistant.config import HomeAssistantConfig
from alphonse.integrations.homeassistant.rest_client import HomeAssistantRestClient
from alphonse.integrations.homeassistant.ws_client import HomeAssistantWsClient

logger = get_component_logger("integrations.homeassistant.adapter")


class HomeAssistantAdapter:
    def __init__(self, config: HomeAssistantConfig) -> None:
        self._config = config
        self._rest = HomeAssistantRestClient(config)
        self._ws = HomeAssistantWsClient(config)
        self._filter = EventFilter(
            allowed_domains=config.allowed_domains,
            allowed_entity_ids=config.allowed_entity_ids,
        )
        self._debouncer = EventDebouncer(config.debounce)
        self._area_cache: dict[str, str] = {}

    def query(self, spec: QuerySpec) -> QueryResult:
        kind = str(spec.kind or "").strip().lower()
        if kind == "states":
            items = self._rest.get_states()
            return QueryResult(ok=True, items=[_normalize_state(item) for item in items])
        if kind == "state":
            entity_id = spec.entity_id or str(spec.filters.get("entity_id") or "").strip()
            if not entity_id:
                return QueryResult(ok=False, error_code="entity_id_required", error_detail="entity_id is required")
            item = self._rest.get_state(entity_id)
            if item is None:
                return QueryResult(ok=True, item=None)
            return QueryResult(ok=True, item=_normalize_state(item))
        return QueryResult(ok=False, error_code="unsupported_query_kind", error_detail=f"kind={kind}")

    def execute(self, action_request: ActionRequest) -> ActionResult:
        if action_request.action_type != "call_service":
            return ActionResult(
                transport_ok=False,
                effect_applied_ok=None,
                readback_performed=False,
                readback_state=None,
                error_code="unsupported_action_type",
                error_detail=f"action_type={action_request.action_type}",
            )
        try:
            self._rest.call_service(
                domain=action_request.domain,
                service=action_request.service,
                data=action_request.data,
                target=action_request.target,
            )
        except Exception as exc:
            return ActionResult(
                transport_ok=False,
                effect_applied_ok=None,
                readback_performed=False,
                readback_state=None,
                error_code="transport_error",
                error_detail=str(exc),
            )

        if not action_request.readback:
            return ActionResult(
                transport_ok=True,
                effect_applied_ok=None,
                readback_performed=False,
                readback_state=None,
            )

        entity_id = _resolve_readback_entity_id(action_request)
        if not entity_id:
            return ActionResult(
                transport_ok=True,
                effect_applied_ok=None,
                readback_performed=False,
                readback_state=None,
            )

        readback_state = self._rest.get_state(entity_id)
        if readback_state is None:
            return ActionResult(
                transport_ok=True,
                effect_applied_ok=False,
                readback_performed=True,
                readback_state=None,
            )

        effect = _matches_expectations(readback_state, action_request)
        return ActionResult(
            transport_ok=True,
            effect_applied_ok=effect,
            readback_performed=True,
            readback_state=_normalize_state(readback_state),
        )

    def subscribe(
        self,
        spec: SubscribeSpec,
        on_event: Callable[[NormalizedEvent], None],
    ) -> SubscriptionHandle:
        event_type = str(spec.event_type or "state_changed").strip() or "state_changed"

        def _on_raw_event(raw_event: dict[str, Any]) -> None:
            if str(raw_event.get("event_type") or "") != event_type:
                return
            data = raw_event.get("data") if isinstance(raw_event.get("data"), dict) else {}
            new_state = data.get("new_state") if isinstance(data.get("new_state"), dict) else {}
            entity_id = str(new_state.get("entity_id") or data.get("entity_id") or "").strip() or None
            domain = _extract_domain(entity_id)
            if not self._filter.allows(domain=domain, entity_id=entity_id):
                return
            if self._debouncer.is_suppressed(raw_event):
                return
            area_id = self._resolve_area_id(entity_id=entity_id, raw_event=raw_event)
            payload = _normalize_state_change_event(raw_event, area_id=area_id)
            on_event(payload)

        sub_id = self._ws.subscribe_events(event_type=event_type, callback=_on_raw_event)

        def _unsubscribe() -> None:
            self._ws.unsubscribe(sub_id)

        return SubscriptionHandle(subscription_id=sub_id, unsubscribe=_unsubscribe)

    def stop(self) -> None:
        self._ws.stop()

    def _resolve_area_id(self, *, entity_id: str | None, raw_event: dict[str, Any]) -> str | None:
        if not entity_id:
            return None
        data = raw_event.get("data") if isinstance(raw_event.get("data"), dict) else {}
        new_state = data.get("new_state") if isinstance(data.get("new_state"), dict) else {}
        attrs = new_state.get("attributes") if isinstance(new_state.get("attributes"), dict) else {}
        candidate = attrs.get("area_id")
        if candidate:
            resolved = str(candidate)
            self._area_cache[entity_id] = resolved
            return resolved
        return self._area_cache.get(entity_id)


def _resolve_readback_entity_id(action_request: ActionRequest) -> str | None:
    if action_request.readback_entity_id:
        return action_request.readback_entity_id
    target = action_request.target if isinstance(action_request.target, dict) else {}
    entity_id = target.get("entity_id")
    if isinstance(entity_id, list):
        if not entity_id:
            return None
        return str(entity_id[0])
    if entity_id is None:
        return None
    return str(entity_id)


def _matches_expectations(state: dict[str, Any], action: ActionRequest) -> bool | None:
    expected_state = str(action.expected_state).strip() if action.expected_state is not None else None
    attrs = state.get("attributes") if isinstance(state.get("attributes"), dict) else {}

    if expected_state is not None:
        current_state = str(state.get("state")) if state.get("state") is not None else None
        if current_state != expected_state:
            return False

    if action.expected_attributes:
        for key, value in action.expected_attributes.items():
            if attrs.get(key) != value:
                return False

    if expected_state is None and not action.expected_attributes:
        return None
    return True


def _normalize_state(item: dict[str, Any]) -> dict[str, Any]:
    entity_id = str(item.get("entity_id") or "").strip() or None
    attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
    return {
        "entity_id": entity_id,
        "domain": _extract_domain(entity_id),
        "state": item.get("state"),
        "attributes": attrs,
        "last_changed": item.get("last_changed"),
        "last_updated": item.get("last_updated"),
        "area_id": attrs.get("area_id"),
    }


def _normalize_state_change_event(event: dict[str, Any], *, area_id: str | None) -> NormalizedEvent:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    old_state = data.get("old_state") if isinstance(data.get("old_state"), dict) else {}
    new_state = data.get("new_state") if isinstance(data.get("new_state"), dict) else {}
    entity_id = str(new_state.get("entity_id") or data.get("entity_id") or "").strip() or None
    domain = _extract_domain(entity_id)

    return NormalizedEvent(
        event_type=str(event.get("event_type") or "state_changed"),
        entity_id=entity_id,
        domain=domain,
        area_id=area_id,
        old_state=str(old_state.get("state")) if old_state.get("state") is not None else None,
        new_state=str(new_state.get("state")) if new_state.get("state") is not None else None,
        attributes=(new_state.get("attributes") if isinstance(new_state.get("attributes"), dict) else {}),
        changed_at=(
            str(new_state.get("last_changed"))
            if new_state.get("last_changed") is not None
            else str(event.get("time_fired"))
            if event.get("time_fired") is not None
            else None
        ),
        raw_event=event,
    )


def _extract_domain(entity_id: str | None) -> str | None:
    if not entity_id or "." not in entity_id:
        return None
    return entity_id.split(".", 1)[0]
