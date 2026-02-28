from __future__ import annotations

from pathlib import Path

from alphonse.agent.cognition.memory import MemoryService
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
    assert mission_res["status"] == "ok"
    assert isinstance(mission_res["result"], dict)
    assert mission_res["result"]["mission"]["mission_id"] == "m1"

    active_tool = ListActiveMissionsTool()
    active_res = active_tool.execute(user_id="alex")
    assert active_res["status"] == "ok"
    assert active_res["result"]["count"] >= 1

    pointer_tool = GetWorkspacePointerTool()
    pointer_res = pointer_tool.execute(key="leads_db_path", user_id="alex")
    assert pointer_res["status"] == "ok"
    assert pointer_res["result"]["value"] == "/tmp/leads.db"

    search_tool = SearchEpisodesTool()
    search_res = search_tool.execute(query="Guadalajara", user_id="alex", mission_id="m1")
    assert search_res["status"] == "ok"
    assert search_res["result"]["count"] >= 1
