from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.cognition.memory import record_after_tool_call
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent import identity as users_store
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry import planner_tool_schemas
from alphonse.agent.tools.memory_tools import GetMissionTool
from alphonse.agent.tools.memory_tools import GetWorkspacePointerTool
from alphonse.agent.tools.memory_tools import ListActiveMissionsTool
from alphonse.agent.tools.memory_tools import SearchEpisodesTool
from alphonse.agent.tools.memory_tools import SearchSummariesTool


def test_memory_read_tools_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "alex", "display_name": "Alex", "is_active": True})
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
    assert "memory.search_summaries" in keys
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
    users_store.upsert_user({"user_id": "user_a", "display_name": "User A", "is_active": True})
    users_store.upsert_user({"user_id": "user_b", "display_name": "User B", "is_active": True})
    registry = build_default_tool_registry()

    upsert = registry.get("memory.upsert_operational_fact")
    assert upsert is not None
    created = upsert.invoke(
        {
            "state": {"actor_person_id": "user_a"},
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
            "state": {"actor_person_id": "user_a"},
            "query": "payroll",
        }
    )
    assert own_search["exception"] is None
    assert own_search["output"]["count"] == 1

    other_search = search.invoke(
        {
            "state": {"actor_person_id": "user_b"},
            "query": "payroll",
        }
    )
    assert other_search["exception"] is None
    assert other_search["output"]["count"] == 0

    remove = registry.get("memory.remove_operational_fact")
    assert remove is not None
    denied_delete = remove.invoke(
        {
            "state": {"actor_person_id": "user_b"},
            "key": "ops.payroll.cutoff",
        }
    )
    assert denied_delete["exception"] is None
    assert denied_delete["output"]["deleted"] is False

    allowed_delete = remove.invoke(
        {
            "state": {"actor_person_id": "user_a"},
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
    assert "memory.search_summaries" in schemas
    upsert_params = dict(schemas["memory.upsert_operational_fact"].get("parameters") or {})
    assert upsert_params.get("required") == ["key", "title", "fact_type"]
    for name in (
        "memory.search_episodes",
        "memory.search_summaries",
        "memory.get_mission",
        "memory.list_active_missions",
        "memory.get_workspace",
        "memory.upsert_operational_fact",
        "memory.search_operational_facts",
        "memory.remove_operational_fact",
    ):
        params = dict(schemas[name].get("parameters") or {})
        properties = dict(params.get("properties") or {})
        assert "user_id" not in properties
        assert "created_by" not in properties


def test_search_episodes_does_not_read_legacy_alias_bucket(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "canonical_user", "display_name": "Canonical User", "is_active": True})
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
            "actor_person_id": "canonical_user",
            "channel_target": "legacy_chat_42",
        },
    )
    assert result["exception"] is None
    output = result["output"]
    assert output["count"] == 0


def test_record_hook_and_tools_share_canonical_owner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "user_canon", "display_name": "Canon", "is_active": True})
    record_after_tool_call(
        task_record=TaskRecord(goal="find client info", task_id="task_1", status="running", user_id="user_canon"),
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
            "actor_person_id": "user_canon",
            "incoming_user_id": "user_legacy",
            "channel_target": "telegram:legacy",
        },
    )
    assert search_res["output"]["count"] >= 1


def test_search_episodes_output_is_domain_agnostic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "user_a", "display_name": "User A", "is_active": True})
    service = MemoryService()
    service.append_episode(
        user_id="user_a",
        mission_id="m_a",
        event_type="web_research",
        payload={"result": "Potential client discovered."},
    )
    search_tool = SearchEpisodesTool()
    found = search_tool.execute(
        query="Potential client discovered",
        state={"actor_person_id": "user_a"},
    )
    output = found["output"]
    assert found["exception"] is None
    assert int(output["count"]) >= 1
    assert "crm_status" not in output
    assert "crm_count" not in output
    assert "crm_facts" not in output


def test_memory_tools_fail_without_canonical_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    search_tool = SearchEpisodesTool()
    result = search_tool.execute(query="anything", state={"incoming_user_id": "legacy_chat_42"})
    assert str((result.get("exception") or {}).get("code") or "") == "missing_user_id"


def test_search_summaries_finds_monthly_summary_for_canonical_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "canonical_alex", "display_name": "Alex", "is_active": True})
    summary_path = tmp_path / "memory" / "canonical_alex" / "summaries" / "monthly" / "2026" / "month_2026-04.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("April contained a major memory refactor.\n", encoding="utf-8")

    tool = SearchSummariesTool()
    result = tool.execute(
        query="April",
        period_kind="monthly",
        state={"actor_person_id": "canonical_alex"},
    )

    assert result["exception"] is None
    output = result["output"]
    assert output["user_id"] == "canonical_alex"
    assert output["count"] == 1
    hit = output["hits"][0]
    assert hit["period_kind"] == "monthly"
    assert hit["date_range"] == {"start": "2026-04-01", "end": "2026-04-30"}


def test_search_summaries_filters_period_and_user_scope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "user_a", "display_name": "User A", "is_active": True})
    users_store.upsert_user({"user_id": "user_b", "display_name": "User B", "is_active": True})
    daily = tmp_path / "memory" / "user_a" / "summaries" / "daily" / "2026" / "day_2026-04-10.md"
    weekly = tmp_path / "memory" / "user_b" / "summaries" / "weekly" / "2026" / "week_2026-04-06_W15.md"
    daily.parent.mkdir(parents=True, exist_ok=True)
    weekly.parent.mkdir(parents=True, exist_ok=True)
    daily.write_text("shared keyword in daily summary\n", encoding="utf-8")
    weekly.write_text("shared keyword in weekly summary\n", encoding="utf-8")

    tool = SearchSummariesTool()
    result = tool.execute(query="shared keyword", period_kind="weekly", state={"actor_person_id": "user_a"})

    assert result["exception"] is None
    assert result["output"]["count"] == 0


def test_search_summaries_requires_canonical_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    result = SearchSummariesTool().execute(query="anything", state={"incoming_user_id": "legacy_chat_42"})

    assert str((result.get("exception") or {}).get("code") or "") == "missing_user_id"


def test_memory_search_prefers_channel_mapping_over_stale_owner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "canonical_alex", "display_name": "Alex", "is_active": True})
    users_store.upsert_user({"user_id": "owner-1", "display_name": "Stale Owner", "is_active": True})
    users_store.upsert_service_user_id(user_id="canonical_alex", service_id=2, service_user_id="8553589429")
    service = MemoryService()
    service.append_episode(
        user_id="canonical_alex",
        mission_id="m_canonical",
        event_type="note",
        payload={"result": "canonical-only-memory"},
    )
    service.append_episode(
        user_id="owner-1",
        mission_id="m_stale",
        event_type="note",
        payload={"result": "stale-owner-memory"},
    )

    result = SearchEpisodesTool().execute(
        query="canonical-only-memory",
        state={
            "owner_id": "owner-1",
            "actor_person_id": "owner-1",
            "channel_type": "telegram",
            "channel_target": "8553589429",
        },
    )

    assert result["exception"] is None
    assert result["output"]["user_id"] == "canonical_alex"
    assert result["output"]["count"] == 1


def test_memory_write_prefers_channel_mapping_over_stale_task_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "canonical_alex", "display_name": "Alex", "is_active": True})
    users_store.upsert_user({"user_id": "owner-1", "display_name": "Stale Owner", "is_active": True})
    users_store.upsert_service_user_id(user_id="canonical_alex", service_id=2, service_user_id="8553589429")
    task_record = TaskRecord(goal="write canonical memory", task_id="task-write-1", status="running", user_id="owner-1")
    task_record.append_fact("channel_type: telegram")
    task_record.append_fact("channel_target: 8553589429")

    record_after_tool_call(
        task_record=task_record,
        current={"step_id": "step_1"},
        tool_name="get_time",
        args={},
        result={"output": {"now": "2026-05-17T12:00:00Z"}, "exception": None, "metadata": {}},
        correlation_id="corr-canonical-write",
    )

    assert (tmp_path / "memory" / "canonical_alex" / "episodic").exists()
    assert not (tmp_path / "memory" / "owner-1" / "episodic").exists()
