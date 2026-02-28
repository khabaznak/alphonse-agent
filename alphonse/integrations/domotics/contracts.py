from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class QuerySpec:
    kind: str
    entity_id: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    ok: bool
    items: list[dict[str, Any]] = field(default_factory=list)
    item: dict[str, Any] | None = None
    error_code: str | None = None
    error_detail: str | None = None


@dataclass(frozen=True)
class ActionRequest:
    action_type: str = "call_service"
    domain: str = ""
    service: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] | None = None
    readback: bool = True
    readback_entity_id: str | None = None
    expected_state: str | None = None
    expected_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    transport_ok: bool
    effect_applied_ok: bool | None
    readback_performed: bool
    readback_state: dict[str, Any] | None
    error_code: str | None = None
    error_detail: str | None = None


@dataclass(frozen=True)
class SubscribeSpec:
    event_type: str = "state_changed"
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedEvent:
    event_type: str
    entity_id: str | None
    domain: str | None
    area_id: str | None
    old_state: str | None
    new_state: str | None
    attributes: dict[str, Any]
    changed_at: str | None
    raw_event: dict[str, Any]


@dataclass(frozen=True)
class SubscriptionHandle:
    subscription_id: str
    unsubscribe: Callable[[], None]


class DomoticsAdapter(Protocol):
    def query(self, spec: QuerySpec) -> QueryResult:
        ...

    def execute(self, action_request: ActionRequest) -> ActionResult:
        ...

    def subscribe(
        self,
        spec: SubscribeSpec,
        on_event: Callable[[NormalizedEvent], None],
    ) -> SubscriptionHandle:
        ...

    def stop(self) -> None:
        ...
