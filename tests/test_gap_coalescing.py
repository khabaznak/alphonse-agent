from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.capability_gaps.coalescing import coalesce_open_intent_gaps
from alphonse.agent.nervous_system.capability_gaps import insert_gap
from alphonse.agent.nervous_system.gap_proposals import list_gap_proposals
from alphonse.agent.nervous_system.migrate import apply_schema


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_coalescer_creates_pending_proposal_for_repeated_proposed_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    insert_gap(
        {
            "user_text": "Que horas son?",
            "reason": "proposed_intent_unmapped",
            "status": "open",
            "metadata": {
                "proposed_intent": "time",
                "proposed_intent_aliases": ["hora", "current time"],
                "proposed_intent_confidence": 0.9,
            },
        }
    )
    insert_gap(
        {
            "user_text": "What time is it?",
            "reason": "proposed_intent_unmapped",
            "status": "open",
            "metadata": {
                "proposed_intent": "time",
                "proposed_intent_aliases": ["clock check"],
                "proposed_intent_confidence": 0.8,
            },
        }
    )

    created = coalesce_open_intent_gaps(limit=100, min_cluster_size=2)
    assert len(created) == 1

    proposals = list_gap_proposals(status="pending", limit=20)
    assert proposals
    assert proposals[0].get("proposed_intent_name") == "time"
    assert proposals[0].get("proposed_next_action") == "plan"
    assert float(proposals[0].get("confidence") or 0) >= 0.8


def test_coalescer_does_not_duplicate_existing_pending_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    insert_gap(
        {
            "user_text": "What time is it?",
            "reason": "proposed_intent_unmapped",
            "status": "open",
            "metadata": {
                "proposed_intent": "time",
                "proposed_intent_aliases": ["current time"],
                "proposed_intent_confidence": 0.9,
            },
        }
    )
    insert_gap(
        {
            "user_text": "Que horas son?",
            "reason": "proposed_intent_unmapped",
            "status": "open",
            "metadata": {
                "proposed_intent": "time",
                "proposed_intent_aliases": ["hora"],
                "proposed_intent_confidence": 0.9,
            },
        }
    )

    first = coalesce_open_intent_gaps(limit=100, min_cluster_size=2)
    second = coalesce_open_intent_gaps(limit=100, min_cluster_size=2)
    assert len(first) == 1
    assert second == []
