from __future__ import annotations

from typing import Callable

from alphonse.integrations.domotics.contracts import (
    ActionRequest,
    ActionResult,
    DomoticsAdapter,
    NormalizedEvent,
    QueryResult,
    QuerySpec,
    SubscribeSpec,
    SubscriptionHandle,
)


class DomoticsFacade:
    def __init__(self, adapter: DomoticsAdapter) -> None:
        self._adapter = adapter

    def query(self, spec: QuerySpec) -> QueryResult:
        return self._adapter.query(spec)

    def execute(self, action_request: ActionRequest) -> ActionResult:
        return self._adapter.execute(action_request)

    def subscribe(
        self,
        spec: SubscribeSpec,
        on_event: Callable[[NormalizedEvent], None],
    ) -> SubscriptionHandle:
        return self._adapter.subscribe(spec, on_event)

    def stop(self) -> None:
        self._adapter.stop()

    @property
    def adapter(self) -> DomoticsAdapter:
        return self._adapter
