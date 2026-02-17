from __future__ import annotations

from pathlib import Path

from alphonse.agent.services.scratchpad_service import ScratchpadService
from alphonse.agent.tools.scratchpad_tools import ScratchpadAppendTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadCreateTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadListTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadReadTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadSearchTool


def _service(tmp_path: Path) -> ScratchpadService:
    return ScratchpadService(root=tmp_path / "data" / "scratchpad")


def test_tools_return_canonical_shape(tmp_path: Path) -> None:
    service = _service(tmp_path)
    create = ScratchpadCreateTool(service)
    append = ScratchpadAppendTool(service)
    read = ScratchpadReadTool(service)
    created = create.execute(title="Weekly chores", scope="household", tags=["chores"], user_id="u1")
    assert created["status"] == "ok"
    assert "result" in created
    assert "error" in created
    assert "metadata" in created
    doc_id = str(created["result"]["doc_id"])
    appended = append.execute(doc_id=doc_id, text="Take out trash", user_id="u1")
    assert appended["status"] == "ok"
    fetched = read.execute(doc_id=doc_id, mode="tail", max_chars=6000, user_id="u1")
    assert fetched["status"] == "ok"
    assert fetched["result"]["doc_id"] == doc_id


def test_tools_respect_user_scope(tmp_path: Path) -> None:
    service = _service(tmp_path)
    create = ScratchpadCreateTool(service)
    read = ScratchpadReadTool(service)
    created = create.execute(title="Private note", user_id="user_a")
    doc_id = str(created["result"]["doc_id"])
    denied = read.execute(doc_id=doc_id, user_id="user_b")
    assert denied["status"] == "failed"
    assert isinstance(denied["error"], dict)
    assert denied["error"]["code"] == "doc_not_found"


def test_list_and_search_tools(tmp_path: Path) -> None:
    service = _service(tmp_path)
    create = ScratchpadCreateTool(service)
    append = ScratchpadAppendTool(service)
    list_tool = ScratchpadListTool(service)
    search = ScratchpadSearchTool(service)
    created = create.execute(title="Weekly chores", scope="household", tags=["family", "chores"], user_id="u1")
    doc_id = str(created["result"]["doc_id"])
    _ = append.execute(doc_id=doc_id, text="Clean kitchen", user_id="u1")
    listed = list_tool.execute(scope="household", user_id="u1")
    assert listed["status"] == "ok"
    assert listed["result"]["docs"]
    hits = search.execute(query="kitchen", scope="household", user_id="u1")
    assert hits["status"] == "ok"
    assert hits["result"]["hits"]

