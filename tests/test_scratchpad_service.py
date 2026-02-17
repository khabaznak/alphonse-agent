from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphonse.agent.services.scratchpad_service import ScratchpadService


def test_index_is_auto_created(tmp_path: Path) -> None:
    service = ScratchpadService(root=tmp_path / "data" / "scratchpad")
    result = service.create_doc(user_id="u1", title="Weekly chores")
    assert str(result.get("doc_id") or "").startswith("sp_")
    index_path = tmp_path / "data" / "scratchpad" / "u1" / "index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert isinstance(payload.get("docs"), dict)
    assert result["doc_id"] in payload["docs"]


def test_append_uses_timestamped_separator(tmp_path: Path) -> None:
    service = ScratchpadService(root=tmp_path / "data" / "scratchpad")
    created = service.create_doc(user_id="u1", title="Daily log")
    doc_id = str(created["doc_id"])
    _ = service.append_doc(user_id="u1", doc_id=doc_id, text="Bought groceries.")
    read = service.read_doc(user_id="u1", doc_id=doc_id, mode="full", max_chars=2000)
    content = str(read.get("content") or "")
    assert "\n\n---\n\n### " in content
    assert "Bought groceries." in content


def test_read_modes_are_bounded(tmp_path: Path) -> None:
    service = ScratchpadService(root=tmp_path / "data" / "scratchpad")
    created = service.create_doc(user_id="u1", title="Long note")
    doc_id = str(created["doc_id"])
    _ = service.append_doc(user_id="u1", doc_id=doc_id, text=("A" * 9000))
    tail = service.read_doc(user_id="u1", doc_id=doc_id, mode="tail", max_chars=600)
    head = service.read_doc(user_id="u1", doc_id=doc_id, mode="head", max_chars=600)
    summary = service.read_doc(user_id="u1", doc_id=doc_id, mode="summary", max_chars=600)
    assert len(str(tail.get("content") or "")) <= 600
    assert len(str(head.get("content") or "")) <= 600
    assert len(str(summary.get("content") or "")) <= 600
    assert str(tail.get("mode") or "") == "tail"


def test_path_traversal_is_blocked(tmp_path: Path) -> None:
    service = ScratchpadService(root=tmp_path / "data" / "scratchpad")
    created = service.create_doc(user_id="u1", title="Secure note")
    doc_id = str(created["doc_id"])
    index_path = tmp_path / "data" / "scratchpad" / "u1" / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["docs"][doc_id]["path"] = "../escape.md"
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="path_traversal_blocked"):
        service.read_doc(user_id="u1", doc_id=doc_id)


def test_list_and_search_filters(tmp_path: Path) -> None:
    service = ScratchpadService(root=tmp_path / "data" / "scratchpad")
    chores = service.create_doc(user_id="u1", title="Weekly household chores", scope="household", tags=["chores", "family"])
    project = service.create_doc(user_id="u1", title="API refactor notes", scope="project", tags=["engineering"])
    _ = service.append_doc(user_id="u1", doc_id=str(chores["doc_id"]), text="Laundry and dishes")
    _ = service.append_doc(user_id="u1", doc_id=str(project["doc_id"]), text="Tool registry migration")
    listed = service.list_docs(user_id="u1", scope="household")
    docs = listed.get("docs") if isinstance(listed.get("docs"), list) else []
    assert len(docs) == 1
    assert docs[0]["doc_id"] == chores["doc_id"]
    hits = service.search_docs(user_id="u1", query="chores", scope="household")
    assert hits["hits"]
    assert hits["hits"][0]["doc_id"] == chores["doc_id"]

