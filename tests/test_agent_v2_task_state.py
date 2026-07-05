from __future__ import annotations

from datetime import datetime

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence import TaskState
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_from_message_maps_canonical_core_message_fields() -> None:
    message = CoreMessage(
        timestamp=datetime.now().astimezone(),
        prompt="Build the file",
        user="gaby",
        project_id="home",
        tag="writing",
    )

    state = TaskState.from_message(message, message_id="msg-1")

    assert state.message_id == "msg-1"
    assert state.goal == "Build the file"
    assert state.user == "gaby"
    assert state.project_id == "home"
    assert state.tag == "writing"
    assert state.recent_conversation_md == "- gaby: Build the file"


def test_from_message_carries_message_metadata() -> None:
    message = CoreMessage(
        timestamp=datetime.now().astimezone(),
        prompt="/project new",
        user="alex",
        metadata={"is_command": True, "command": "project", "command_args": "new"},
    )

    state = TaskState.from_message(message)

    assert state.metadata == {
        "is_command": True,
        "command": "project",
        "command_args": "new",
    }


def test_from_queued_message_preserves_queue_and_message_metadata() -> None:
    queue = InMemoryMessageQueue()
    queued = CommunicationChannel(queue).queue_message(
        prompt="/project new",
        user="alex",
        project_id="alpha",
        tag="work",
    )

    state = TaskState.from_queued_message(queued)

    assert state.message_id == queued.message_id
    assert state.goal == "/project new"
    assert state.user == "alex"
    assert state.project_id == "alpha"
    assert state.tag == "work"
    assert state.metadata == {
        "is_command": True,
        "command": "project",
        "command_args": "new",
    }


def test_markdown_defaults_are_none_bullets() -> None:
    state = TaskState()

    assert state.facts_md == "- (none)"
    assert state.recent_conversation_md == "- (none)"
    assert state.plan_md == "- (none)"
    assert state.acceptance_criteria_md == "- (none)"
    assert state.memory_facts_md == "- (none)"
    assert state.tool_call_history_md == "- (none)"
    assert state.updates_md == "- (none)"


def test_append_helpers_use_bullet_markdown() -> None:
    state = TaskState()

    state.append_fact("fact one")
    state.append_fact("- fact two")
    state.append_plan_line("plan one")
    state.append_acceptance_criterion("criterion one")
    state.append_memory_fact("memory one")
    state.append_tool_call_history_entry("tool one")
    state.append_recent_conversation_line("conversation one")
    state.append_update("update one")

    assert state.facts_md == "- fact one\n- fact two"
    assert state.plan_md == "- plan one"
    assert state.acceptance_criteria_md == "- criterion one"
    assert state.memory_facts_md == "- memory one"
    assert state.tool_call_history_md == "- tool one"
    assert state.recent_conversation_md == "- conversation one"
    assert state.updates_md == "- update one"


def test_to_dict_from_dict_round_trip() -> None:
    state = TaskState(
        task_id="task-1",
        message_id="msg-1",
        user="alex",
        project_id="alpha",
        tag="core",
        correlation_id="corr-1",
        goal="Goal",
        status="done",
        outcome={"ok": True},
        check_verdict="wip",
        check_reason="Still working",
        check_confidence=0.4,
        check_evidence_refs=["ref-1"],
        check_new_message_count=2,
        pdca_cycle_count=3,
        metadata={"is_command": False},
    )
    state.append_update("updated")

    restored = TaskState.from_dict(state.to_dict())

    assert restored.to_dict() == state.to_dict()


def test_set_check_result_validates_verdict_and_clamps_confidence() -> None:
    state = TaskState()

    state.set_check_result(
        verdict="MISSION_SUCCESS",
        reason="done",
        confidence=2.5,
        evidence_refs=[" ref-1 ", ""],
        new_message_count=-1,
    )

    assert state.check_verdict == "mission_success"
    assert state.check_reason == "done"
    assert state.check_confidence == 1.0
    assert state.check_evidence_refs == ["ref-1"]
    assert state.check_new_message_count == 0

    with pytest.raises(ValueError, match="invalid_check_verdict"):
        state.set_check_result(verdict="unknown")


def test_replan_clears_goal_acceptance_criteria_status_and_outcome() -> None:
    state = TaskState(
        goal="old",
        acceptance_criteria_md="- pass",
        status="failed",
        outcome={"error": "no"},
    )

    state.replan()

    assert state.goal == ""
    assert state.acceptance_criteria_md == "- (none)"
    assert state.status == "running"
    assert state.outcome is None


def test_to_markdown_prompt_includes_expected_sections_without_mutating() -> None:
    state = TaskState(task_id="task-1", user="alex", goal="Create a project")
    state.append_plan_line("First step")
    before = state.to_dict()

    rendered = state.to_markdown_prompt()

    assert "# Task Metadata" in rendered
    assert "# Goal" in rendered
    assert "Create a project" in rendered
    assert "# Recent Conversation" in rendered
    assert "# Facts" in rendered
    assert "# Current Plan" in rendered
    assert "- First step" in rendered
    assert "# Acceptance Criteria" in rendered
    assert "# Updates" in rendered
    assert "# Memory Facts" in rendered
    assert "# Tool Call History" in rendered
    assert "# Check Result" in rendered
    assert "# Outcome" in rendered
    assert state.to_dict() == before
