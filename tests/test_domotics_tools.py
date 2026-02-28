from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.tools.domotics_tools import (
    DomoticsExecuteTool,
    DomoticsQueryTool,
    DomoticsSubscribeTool,
)
from alphonse.integrations.domotics.contracts import (
    ActionResult,
    NormalizedEvent,
    QueryResult,
    SubscriptionHandle,
)


@dataclass
class _FakeFacade:
    def query(self, spec):
        if spec.kind == "state":
            return QueryResult(ok=True, item={"entity_id": spec.entity_id, "state": "off"})
        return QueryResult(ok=True, items=[{"entity_id": "light.kitchen", "state": "on"}])

    def execute(self, _request):
        return ActionResult(
            transport_ok=True,
            effect_applied_ok=True,
            readback_performed=True,
            readback_state={"entity_id": "light.kitchen", "state": "on"},
        )

    def subscribe(self, _spec, on_event):
        on_event(
            NormalizedEvent(
                event_type="state_changed",
                entity_id="light.kitchen",
                domain="light",
                area_id=None,
                old_state="off",
                new_state="on",
                attributes={},
                changed_at="2026-02-28T00:00:00Z",
                raw_event={"event_type": "state_changed"},
            )
        )
        return SubscriptionHandle(subscription_id="sub-1", unsubscribe=lambda: None)


def test_domotics_query_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphonse.agent.tools.domotics_tools.get_domotics_facade",
        lambda: _FakeFacade(),
    )

    tool = DomoticsQueryTool()
    result = tool.execute(kind="state", entity_id="light.kitchen")

    assert result["status"] == "ok"
    assert (result["result"] or {}).get("item", {}).get("entity_id") == "light.kitchen"


def test_domotics_execute_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphonse.agent.tools.domotics_tools.get_domotics_facade",
        lambda: _FakeFacade(),
    )

    tool = DomoticsExecuteTool()
    result = tool.execute(domain="light", service="turn_on", target={"entity_id": "light.kitchen"})

    assert result["status"] == "ok"
    assert (result["result"] or {}).get("transport_ok") is True


def test_domotics_subscribe_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphonse.agent.tools.domotics_tools.get_domotics_facade",
        lambda: _FakeFacade(),
    )

    tool = DomoticsSubscribeTool()
    result = tool.execute(duration_seconds=0.5)

    assert result["status"] == "ok"
    assert (result["result"] or {}).get("event_count", 0) >= 1
