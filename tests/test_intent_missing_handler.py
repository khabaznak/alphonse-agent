from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import IntentCatalogService, IntentCatalogStore, IntentSpec
from alphonse.agent.cortex import graph as cortex_graph
from alphonse.agent.nervous_system.migrate import apply_schema


def _setup_service(tmp_path: Path) -> IntentCatalogService:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    store = IntentCatalogStore(str(db_path))
    spec = IntentSpec(
        intent_name="custom.intent",
        category="task_plane",
        description="Custom intent",
        examples=["custom intent"],
        required_slots=[],
        optional_slots=[],
        default_mode="aventurizacion",
        risk_level="low",
        handler="custom.intent",
        enabled=True,
        intent_version="1.0.0",
        origin="factory",
    )
    store.upsert(spec)
    return IntentCatalogService(store=store, ttl_seconds=0)


def test_missing_handler_creates_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _setup_service(tmp_path)
    monkeypatch.setattr(cortex_graph, "get_catalog_service", lambda: service)
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "last_user_message": "custom intent",
        "intent": "custom.intent",
        "intent_confidence": 0.8,
        "intent_category": "task_plane",
        "routing_needs_clarification": False,
        "plans": [],
    }
    result = cortex_graph._respond_node(state)
    plans = result.get("plans") or []
    assert plans, "Expected a capability gap plan for missing handler"
    assert plans[0].get("plan_type") == "CAPABILITY_GAP"
