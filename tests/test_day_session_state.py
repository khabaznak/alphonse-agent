from __future__ import annotations

from datetime import datetime
from pathlib import Path

from alphonse.agent.session.day_state import build_next_session_state
from alphonse.agent.session.day_state import commit_session_state
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
        user_message="Please run getTime again",
        assistant_message="It is 11:00 now.",
        ability_state={"kind": "tool_calls", "steps": [{"idx": 0, "tool": "getTime", "status": "executed"}]},
        task_state=None,
        planning_context={"facts": {"tool_results": [{"step": 0, "tool": "getTime", "time": "2026-02-15T11:00:00"}]}},
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
    assert any("Please run getTime again" in item for item in reloaded["working_set"])
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
        "last_action": {"tool": "getTime", "summary": "Fetched current time.", "ts": "2026-02-15T11:00:00Z"},
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


def test_session_last_action_can_come_from_task_state_steps() -> None:
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
    task_state = {
        "plan": {
            "steps": [
                {
                    "step_id": "step_1",
                    "status": "executed",
                    "proposal": {"kind": "call_tool", "tool_name": "getTime", "args": {}},
                },
                {
                    "step_id": "step_2",
                    "status": "executed",
                    "proposal": {
                        "kind": "call_tool",
                        "tool_name": "local_audio_output.speak",
                        "args": {"text": "Son las 5:22 p. m."},
                    },
                },
            ],
            "current_step_id": "step_2",
        },
        "facts": {
            "step_2": {
                "tool": "local_audio_output.speak",
                "result": {"status": "ok"},
            }
        },
    }
    updated = build_next_session_state(
        previous=previous,
        channel="telegram",
        user_message="otra vez",
        assistant_message="Listo",
        ability_state=None,
        task_state=task_state,
        planning_context=None,
        pending_interaction=None,
    )
    assert isinstance(updated.get("last_action"), dict)
    assert updated["last_action"]["tool"] == "local_audio_output.speak"
