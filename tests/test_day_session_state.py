from __future__ import annotations

from datetime import datetime
from pathlib import Path

from alphonse.agent.session.day_state import build_next_session_state
from alphonse.agent.session.day_state import commit_session_state
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.agent.session.day_state import render_session_markdown
from alphonse.agent.session.day_state import render_session_prompt_block
from alphonse.agent.session.day_state import resolve_day_session


def test_session_router_maps_same_user_same_day_across_channels(tmp_path: Path) -> None:
    now = datetime.fromisoformat("2026-02-15T10:00:00-06:00")
    first = resolve_day_session(
        user_id="person-123",
        channel="webui",
        timezone_name="America/Mexico_City",
        now=now,
        root_dir=tmp_path,
    )
    second = resolve_day_session(
        user_id="person-123",
        channel="telegram",
        timezone_name="America/Mexico_City",
        now=now,
        root_dir=tmp_path,
    )

    assert first["session_id"] == second["session_id"]
    assert second["channels_seen"] == ["telegram"]


def test_session_state_persists_rev_and_working_set(tmp_path: Path) -> None:
    now = datetime.fromisoformat("2026-02-15T11:00:00-06:00")
    state = resolve_day_session(
        user_id="person-123",
        channel="webui",
        timezone_name="America/Mexico_City",
        now=now,
        root_dir=tmp_path,
    )
    updated = build_next_session_state(
        previous=state,
        channel="webui",
        user_message="Please run get_time again",
        assistant_message="It is 11:00 now.",
        task_record=None,
        pending_interaction=None,
    )
    commit_session_state(updated, root_dir=tmp_path)

    reloaded = resolve_day_session(
        user_id="person-123",
        channel="telegram",
        timezone_name="America/Mexico_City",
        now=now,
        root_dir=tmp_path,
    )

    assert reloaded["rev"] == 1
    assert any("Please run get_time again" in item for item in reloaded["working_set"])
    assert "webui" in reloaded["channels_seen"]
    assert "telegram" in reloaded["channels_seen"]


def test_session_prompt_block_is_bounded() -> None:
    state = {
        "session_id": "u-1|2026-02-15",
        "user_id": "u-1",
        "date": "2026-02-15",
        "rev": 9,
        "channels_seen": ["webui", "telegram", "cli", "api", "x", "y", "z"],
        "working_set": [f"working-{idx}" for idx in range(20)],
        "open_loops": [f"loop-{idx}" for idx in range(20)],
        "last_action": {"tool": "get_time", "summary": "Fetched current time.", "ts": "2026-02-15T11:00:00Z"},
    }
    block = render_session_prompt_block(state)
    assert block.count("\n") + 1 <= 30
    assert "SESSION_STATE (u-1|2026-02-15)" in block
    assert "SESSION_STATE is authoritative working memory" in block


def test_session_markdown_render_matches_json_content() -> None:
    state = {
        "session_id": "u-2|2026-02-15",
        "user_id": "u-2",
        "date": "2026-02-15",
        "rev": 2,
        "channels_seen": ["webui", "telegram"],
        "working_set": ["User asked for reminder", "Assistant confirmed schedule"],
        "open_loops": ["Waiting for exact time"],
        "last_action": {"tool": "createReminder", "summary": "Created a reminder.", "ts": "2026-02-15T12:00:00Z"},
    }

    markdown = render_session_markdown(state)
    assert "Session State: u-2|2026-02-15" in markdown
    assert "- User ID: u-2" in markdown
    assert "- Rev: 2" in markdown
    assert "- User asked for reminder" in markdown
    assert "- Waiting for exact time" in markdown
    assert "- Tool: createReminder" in markdown


def test_session_last_action_can_come_from_task_record_history() -> None:
    previous = {
        "session_id": "u-9|2026-02-15",
        "user_id": "u-9",
        "date": "2026-02-15",
        "rev": 0,
        "channels_seen": ["telegram"],
        "working_set": [],
        "open_loops": [],
        "last_action": None,
    }
    task_record = {
        "task_id": "task-1",
        "tool_call_history_md": (
            "- get_time args={} output={\"time\":\"2026-02-15T17:22:00\"} exception=null\n"
            "- audio.speak_local args={\"text\":\"Son las 5:22 p. m.\"} output={\"status\":\"ok\"} exception=null"
        ),
        "plan_md": "- audio.speak_local args={\"text\":\"Son las 5:22 p. m.\"} intent=Play local audio.",
    }
    updated = build_next_session_state(
        previous=previous,
        channel="telegram",
        user_message="otra vez",
        assistant_message="Listo",
        task_record=task_record,
        pending_interaction=None,
    )
    assert isinstance(updated.get("last_action"), dict)
    assert updated["last_action"]["tool"] == "audio.speak_local"


def test_recent_conversation_block_renders_last_twenty_turns() -> None:
    state = {
        "session_id": "u-3|2026-02-15",
        "user_id": "u-3",
        "date": "2026-02-15",
        "rev": 1,
        "channels_seen": ["telegram"],
        "conversation_events": [
            {"role": "user", "text": f"user-{idx}", "timestamp": f"2026-02-15T00:{idx:02d}:00Z"}
            for idx in range(12)
        ]
        + [
            {"role": "assistant", "text": f"assistant-{idx}", "timestamp": f"2026-02-15T01:{idx:02d}:00Z"}
            for idx in range(12)
        ],
        "working_set": [],
        "open_loops": [],
        "last_action": None,
    }
    block = render_recent_conversation_block(state)
    assert "## RECENT CONVERSATION (last 20 turns)" in block
    assert "user-0" not in block
    assert "assistant-0" in block
    assert "user-4" in block
    assert "user-11" in block
    assert "assistant-11" in block


def test_day_state_migrates_legacy_recent_conversation_to_conversation_events() -> None:
    state = {
        "session_id": "u-legacy|2026-02-15",
        "user_id": "u-legacy",
        "date": "2026-02-15",
        "rev": 0,
        "channels_seen": ["telegram"],
        "recent_conversation": [
            {"user": "hello", "assistant": ""},
            {"user": "second", "assistant": "reply"},
        ],
        "working_set": [],
        "open_loops": [],
        "last_action": None,
    }
    markdown = render_session_markdown(state)
    assert "hello" in markdown
    assert "second" in markdown
    assert "reply" in markdown


def test_day_state_persists_full_transcript_text_without_capping(tmp_path: Path) -> None:
    now = datetime.fromisoformat("2026-02-15T11:00:00-06:00")
    long_user = "U" * 400
    long_assistant = "A" * 450
    state = resolve_day_session(
        user_id="person-full",
        channel="telegram",
        timezone_name="America/Mexico_City",
        now=now,
        root_dir=tmp_path,
    )
    updated = build_next_session_state(
        previous=state,
        channel="telegram",
        user_message=long_user,
        assistant_message=long_assistant,
        task_record=None,
        pending_interaction=None,
    )
    commit_session_state(updated, root_dir=tmp_path)
    reloaded = resolve_day_session(
        user_id="person-full",
        channel="telegram",
        timezone_name="America/Mexico_City",
        now=now,
        root_dir=tmp_path,
    )
    events = reloaded.get("conversation_events")
    assert isinstance(events, list)
    assert len(events) == 2
    assert str(events[0].get("text") or "") == long_user
    assert str(events[1].get("text") or "") == long_assistant


def test_day_state_does_not_duplicate_user_when_assistant_is_appended_later() -> None:
    state = {
        "session_id": "u-dedupe|2026-02-15",
        "user_id": "u-dedupe",
        "date": "2026-02-15",
        "rev": 0,
        "channels_seen": ["telegram"],
        "conversation_events": [],
        "working_set": [],
        "open_loops": [],
        "last_action": None,
    }
    first = build_next_session_state(
        previous=state,
        channel="telegram",
        user_message="hello",
        assistant_message="",
        task_record=None,
        pending_interaction=None,
    )
    second = build_next_session_state(
        previous=first,
        channel="telegram",
        user_message="",
        assistant_message="done",
        task_record=None,
        pending_interaction=None,
    )
    events = second.get("conversation_events")
    assert isinstance(events, list)
    assert len(events) == 2
    assert events[0].get("role") == "user"
    assert events[1].get("role") == "assistant"
