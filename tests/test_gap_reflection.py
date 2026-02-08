from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.capability_gaps.reflection import reflect_gaps
from alphonse.agent.nervous_system.capability_gaps import insert_gap
from alphonse.agent.nervous_system.gap_proposals import list_gap_proposals, update_gap_proposal_status
from alphonse.agent.nervous_system.gap_tasks import list_gap_tasks
from alphonse.agent.nervous_system.migrate import apply_schema


def test_reflect_creates_proposal_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    insert_gap(
        {
            "user_text": "Hi",
            "reason": "unknown_intent",
            "status": "open",
            "channel_type": "cli",
            "channel_id": "cli",
        }
    )

    created = reflect_gaps(limit=10)
    assert len(created) == 1

    proposals = list_gap_proposals(status="pending", limit=10)
    assert proposals
    # Heuristic triage is intentionally disabled; reflection may leave intent unset.
    assert proposals[0].get("proposed_category") == "intent_missing"

    tasks = list_gap_tasks(status="open", limit=10)
    assert tasks == []

    update_gap_proposal_status(proposals[0]["id"], "approved", reviewer="human")
    tasks_after = list_gap_tasks(status="open", limit=10)
    assert tasks_after == []
