from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.cognition.memory import record_after_tool_call
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry import planner_tool_schemas
from alphonse.agent.tools.memory_tools import GetMissionTool
from alphonse.agent.tools.memory_tools import GetWorkspacePointerTool
from alphonse.agent.tools.memory_tools import ListActiveMissionsTool
from alphonse.agent.tools.memory_tools import SearchEpisodesTool


def test_memory_read_tools_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    service = MemoryService()
    service.mission_upsert("alex", "m1", status="active", title="Mission One")
    service.upsert_workspace_pointer("alex", "leads_db_path", "/tmp/leads.db")
    service.append_episode(
        user_id="alex",
        mission_id="m1",
        event_type="web_research",
        payload={"result": "found leads in Guadalajara"},
    )

    mission_tool = GetMissionTool()
    mission_res = mission_tool.execute(mission_id="m1", user_id="alex")
    assert mission_res["exception"] is None
    assert isinstance(mission_res["output"], dict)
    assert mission_res["output"]["mission"]["mission_id"] == "m1"

    active_tool = ListActiveMissionsTool()
    active_res = active_tool.execute(user_id="alex")
    assert active_res["exception"] is None
    assert active_res["output"]["count"] >= 1

    pointer_tool = GetWorkspacePointerTool()
    pointer_res = pointer_tool.execute(key="leads_db_path", user_id="alex")
    assert pointer_res["exception"] is None
    assert pointer_res["output"]["value"] == "/tmp/leads.db"

    search_tool = SearchEpisodesTool()
    search_res = search_tool.execute(query="Guadalajara", user_id="alex", mission_id="m1")
    assert search_res["exception"] is None
    assert search_res["output"]["count"] >= 1


def test_memory_tools_surface_is_read_only() -> None:
    registry = build_default_tool_registry()
    keys = set(registry.keys())
    assert "memory.search_episodes" in keys
    assert "memory.get_mission" in keys
    assert "memory.list_active_missions" in keys
    assert "memory.get_workspace" in keys
    assert "memory.upsert_operational_fact" in keys
    assert "memory.search_operational_facts" in keys
    assert "memory.remove_operational_fact" in keys
    assert "append_episode" not in keys
    assert "mission_upsert" not in keys
    assert "put_artifact" not in keys


def test_operational_fact_tools_execute_end_to_end(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    registry = build_default_tool_registry()

    upsert = registry.get("memory.upsert_operational_fact")
    assert upsert is not None
    created = upsert.invoke(
        {
            "state": {"incoming_user_id": "user_a"},
            "key": "ops.payroll.cutoff",
            "title": "Payroll cutoff",
            "fact_type": "procedure",
            "summary": "Close payroll at 16:00 every Friday.",
            "scope": "private",
            "tags": ["finance", "weekly"],
        }
    )
    assert created["exception"] is None
    created_fact = created["output"]["fact"]
    assert created_fact["key"] == "ops.payroll.cutoff"
    assert created_fact["created_by"] == "user_a"
    assert created_fact["scope"] == "private"

    search = registry.get("memory.search_operational_facts")
    assert search is not None
    own_search = search.invoke(
        {
            "state": {"incoming_user_id": "user_a"},
            "query": "payroll",
        }
    )
    assert own_search["exception"] is None
    assert own_search["output"]["count"] == 1

    other_search = search.invoke(
        {
            "state": {"incoming_user_id": "user_b"},
            "query": "payroll",
        }
    )
    assert other_search["exception"] is None
    assert other_search["output"]["count"] == 0

    remove = registry.get("memory.remove_operational_fact")
    assert remove is not None
    denied_delete = remove.invoke(
        {
            "state": {"incoming_user_id": "user_b"},
            "key": "ops.payroll.cutoff",
        }
    )
    assert denied_delete["exception"] is None
    assert denied_delete["output"]["deleted"] is False

    allowed_delete = remove.invoke(
        {
            "state": {"incoming_user_id": "user_a"},
            "key": "ops.payroll.cutoff",
        }
    )
    assert allowed_delete["exception"] is None
    assert allowed_delete["output"]["deleted"] is True


def test_operational_fact_tools_are_exposed_in_planner_schemas() -> None:
    registry = build_default_tool_registry()
    schemas = {
        str(item.get("function", {}).get("name") or ""): item.get("function", {})
        for item in planner_tool_schemas(registry)
        if isinstance(item, dict)
    }
    assert "memory.upsert_operational_fact" in schemas
    assert "memory.search_operational_facts" in schemas
    assert "memory.remove_operational_fact" in schemas
    upsert_params = dict(schemas["memory.upsert_operational_fact"].get("parameters") or {})
    assert upsert_params.get("required") == ["key", "title", "fact_type"]


def test_search_episodes_reads_legacy_alias_bucket(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    service = MemoryService()
    service.append_episode(
        user_id="legacy_chat_42",
        mission_id="m_legacy",
        event_type="web_research",
        payload={"result": "legacy-customer-record"},
    )
    search_tool = SearchEpisodesTool()
    result = search_tool.execute(
        query="legacy-customer-record",
        state={
            "incoming_user_id": "canonical_user",
            "channel_target": "legacy_chat_42",
        },
    )
    assert result["exception"] is None
    output = result["output"]
    assert output["count"] >= 1
    assert "legacy_chat_42" in output.get("owner_aliases", [])


def test_record_hook_and_tools_share_canonical_owner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    record_after_tool_call(
        state={
            "owner_id": "user_canon",
            "incoming_user_id": "user_legacy",
            "channel_target": "telegram:legacy",
        },
        task_state={"goal": "find client info", "task_id": "task_1", "status": "running"},
        current={"step_id": "step_1"},
        tool_name="get_time",
        args={},
        result={"output": {"now": "2026-03-27T12:00:00Z"}, "exception": None, "metadata": {}},
        correlation_id="corr-owner-unified",
    )
    canon_root = tmp_path / "memory" / "user_canon" / "episodic"
    legacy_root = tmp_path / "memory" / "user_legacy" / "episodic"
    assert canon_root.exists()
    assert not legacy_root.exists()

    search_tool = SearchEpisodesTool()
    search_res = search_tool.execute(
        query="status=ok",
        state={
            "owner_id": "user_canon",
            "incoming_user_id": "user_legacy",
            "channel_target": "telegram:legacy",
        },
    )
    assert search_res["output"]["count"] >= 1


def test_search_episodes_client_query_prefers_operational_facts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    registry = build_default_tool_registry()
    upsert = registry.get("memory.upsert_operational_fact")
    assert upsert is not None
    created = upsert.invoke(
        {
            "state": {"incoming_user_id": "user_a"},
            "key": "crm.client.acme_corp",
            "title": "Client: Acme Corp",
            "fact_type": "integration_note",
            "summary": "Potential client discovered.",
            "tags": ["crm", "client"],
            "scope": "private",
        }
    )
    assert created["exception"] is None

    search_tool = SearchEpisodesTool()
    found = search_tool.execute(
        query="Did we store client info for Acme Corp in CRM?",
        state={"incoming_user_id": "user_a"},
    )
    assert found["output"]["crm_status"] == "found"
    assert int(found["output"]["crm_count"]) >= 1

    missing = search_tool.execute(
        query="Did we store client info for Unknown Co in CRM?",
        state={"incoming_user_id": "user_b"},
    )
    assert missing["output"]["crm_status"] == "not_stored_yet"
    assert int(missing["output"]["crm_count"]) == 0
