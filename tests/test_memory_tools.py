from __future__ import annotations

from pathlib import Path

from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.tools.registry import build_default_tool_registry
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
    assert "search_episodes" in keys
    assert "get_mission" in keys
    assert "list_active_missions" in keys
    assert "get_workspace_pointer" in keys
    assert "append_episode" not in keys
    assert "mission_upsert" not in keys
    assert "put_artifact" not in keys
