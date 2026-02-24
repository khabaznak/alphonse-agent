from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_MAX_WORKING_SET = 6
_MAX_OPEN_LOOPS = 6
_MAX_CHANNELS = 6
_MAX_RECENT_TURNS = 20
_MAX_LINE_LENGTH = 256
_NAMED_SECRET_PATTERN = re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\b\s*[:=]\s*\S+")
_TOKEN_SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{10,}\b")
_PATH_PATTERNS = (
    re.compile(r"/Users/[^\s]+"),
    re.compile(r"/home/[^\s]+"),
    re.compile(r"/opt/[^\s]+"),
    re.compile(r"/var/[^\s]+"),
    re.compile(r"/tmp/[^\s]+"),
    re.compile(r"\b[A-Za-z]:\\[^\s]+"),
)


def resolve_day_session(
    *,
    user_id: str,
    channel: str,
    timezone_name: str,
    now: datetime | None = None,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    local_date = current_local_date(timezone_name=timezone_name, now=now)
    existing = load_session_state(user_id=user_id, local_date=local_date, root_dir=root_dir)
    if existing is None:
        existing = default_session_state(user_id=user_id, local_date=local_date)
    existing["channels_seen"] = _merge_channels(existing.get("channels_seen"), channel)
    return existing


def current_local_date(*, timezone_name: str, now: datetime | None = None) -> str:
    tz = ZoneInfo(timezone_name)
    base = now if isinstance(now, datetime) else datetime.now(tz)
    local = base.astimezone(tz) if base.tzinfo else base.replace(tzinfo=tz)
    return local.date().isoformat()


def default_session_state(*, user_id: str, local_date: str) -> dict[str, Any]:
    canonical_user_id = _canonical_user_id(user_id)
    return {
        "session_id": session_id_for_user_day(canonical_user_id, local_date),
        "user_id": canonical_user_id,
        "date": local_date,
        "rev": 0,
        "channels_seen": [],
        "recent_conversation": [],
        "working_set": [],
        "open_loops": [],
        "last_action": None,
    }


def session_id_for_user_day(user_id: str, local_date: str) -> str:
    safe_user = _canonical_user_id(user_id)
    return f"{safe_user}|{local_date}"


def load_session_state(
    *,
    user_id: str,
    local_date: str,
    root_dir: Path | None = None,
) -> dict[str, Any] | None:
    json_path = _session_json_path(user_id=user_id, local_date=local_date, root_dir=root_dir)
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _normalize_state(payload)


def commit_session_state(
    state: dict[str, Any],
    *,
    root_dir: Path | None = None,
) -> None:
    normalized = _normalize_state(state)
    user_id = str(normalized.get("user_id") or "").strip()
    local_date = str(normalized.get("date") or "").strip()
    if not user_id or not local_date:
        raise ValueError("session state requires user_id and date")
    json_path = _session_json_path(user_id=user_id, local_date=local_date, root_dir=root_dir)
    md_path = _session_markdown_path(user_id=user_id, local_date=local_date, root_dir=root_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json_path, json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_write(md_path, render_session_markdown(normalized))


def build_next_session_state(
    *,
    previous: dict[str, Any],
    channel: str,
    user_message: str,
    assistant_message: str,
    ability_state: dict[str, Any] | None,
    task_state: dict[str, Any] | None,
    planning_context: dict[str, Any] | None,
    pending_interaction: dict[str, Any] | None,
) -> dict[str, Any]:
    state = _normalize_state(previous)
    last_action = _infer_last_action(
        ability_state=ability_state,
        task_state=task_state,
        planning_context=planning_context,
    )
    user_line = _sanitize_line(user_message, max_len=100)
    assistant_line = _sanitize_line(assistant_message, max_len=100)

    working_set: list[str] = []
    if user_line:
        working_set.append(f"User asked: {user_line}")
    if assistant_line:
        working_set.append(f"Assistant replied: {assistant_line}")
    if isinstance(last_action, dict) and str(last_action.get("summary") or "").strip():
        working_set.append(f"Last tool action: {last_action['summary']}")
    for item in _normalize_list(state.get("working_set"), limit=_MAX_WORKING_SET):
        if item not in working_set:
            working_set.append(item)
        if len(working_set) >= _MAX_WORKING_SET:
            break

    open_loops: list[str] = []
    if isinstance(pending_interaction, dict):
        pending_key = _sanitize_line(str(pending_interaction.get("key") or ""), max_len=60)
        pending_type = _sanitize_line(str(pending_interaction.get("type") or ""), max_len=40)
        if pending_key:
            open_loops.append(f"Waiting for user input ({pending_key}).")
        elif pending_type:
            open_loops.append(f"Waiting for user input ({pending_type}).")
        else:
            open_loops.append("Waiting for user input.")

    state["channels_seen"] = _merge_channels(state.get("channels_seen"), channel)
    state["recent_conversation"] = _append_recent_turn(
        existing=state.get("recent_conversation"),
        user_text=user_line,
        assistant_text=assistant_line,
    )
    state["working_set"] = working_set[:_MAX_WORKING_SET]
    state["open_loops"] = open_loops[:_MAX_OPEN_LOOPS]
    state["last_action"] = last_action
    state["rev"] = int(state.get("rev") or 0) + 1
    return state


def render_recent_conversation_block(state: dict[str, Any], *, max_turns: int = _MAX_RECENT_TURNS) -> str:
    normalized = _normalize_state(state)
    turns = normalized.get("recent_conversation") if isinstance(normalized.get("recent_conversation"), list) else []
    lines = [f"## RECENT CONVERSATION (last {max_turns} turns)"]
    if not turns:
        lines.append("- (none)")
        return "\n".join(lines)
    for turn in turns[-max_turns:]:
        if not isinstance(turn, dict):
            continue
        user_text = _sanitize_line(str(turn.get("user") or ""), max_len=120)
        assistant_text = _sanitize_line(str(turn.get("assistant") or ""), max_len=120)
        if user_text:
            lines.append(f"- User: {user_text}")
        if assistant_text:
            lines.append(f"- Assistant: {assistant_text}")
    if len(lines) == 1:
        lines.append("- (none)")
    return "\n".join(lines)


def render_session_prompt_block(state: dict[str, Any]) -> str:
    normalized = _normalize_state(state)
    session_id = str(normalized.get("session_id") or "").strip()
    channels = _normalize_list(normalized.get("channels_seen"), limit=_MAX_CHANNELS)
    working_set = _normalize_list(normalized.get("working_set"), limit=_MAX_WORKING_SET)
    open_loops = _normalize_list(normalized.get("open_loops"), limit=_MAX_OPEN_LOOPS)
    last_action = normalized.get("last_action") if isinstance(normalized.get("last_action"), dict) else None
    last_summary = _sanitize_line(str((last_action or {}).get("summary") or ""), max_len=100)

    lines = [
        f"SESSION_STATE ({session_id})",
        (
            "SESSION_STATE is authoritative working memory for this session/day. "
            "Use it to answer follow-up questions about what happened earlier today, "
            "last tools used, and scratchpad values. Do not ignore it."
        ),
        f"- rev: {int(normalized.get('rev') or 0)}",
        f"- channels_seen: {', '.join(channels) if channels else '(none)'}",
        "- working_set:",
    ]
    if working_set:
        lines.extend([f"  - {item}" for item in working_set])
    else:
        lines.append("  - (none)")
    lines.append(f"- last_action: {last_summary or '(none)'}")
    lines.append("- open_loops:")
    if open_loops:
        lines.extend([f"  - {item}" for item in open_loops])
    else:
        lines.append("  - (none)")
    return "\n".join(lines)


def render_session_markdown(state: dict[str, Any]) -> str:
    normalized = _normalize_state(state)
    lines = [
        f"# Session State: {normalized['session_id']}",
        "",
        f"- User ID: {normalized['user_id']}",
        f"- Date: {normalized['date']}",
        f"- Rev: {normalized['rev']}",
        f"- Channels Seen: {', '.join(normalized['channels_seen']) if normalized['channels_seen'] else '(none)'}",
        "",
        "## Recent Conversation",
    ]
    recent_turns = normalized.get("recent_conversation") if isinstance(normalized.get("recent_conversation"), list) else []
    if recent_turns:
        for turn in recent_turns:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("user") or "").strip():
                lines.append(f"- User: {turn.get('user')}")
            if str(turn.get("assistant") or "").strip():
                lines.append(f"- Assistant: {turn.get('assistant')}")
    else:
        lines.append("- (none)")
    lines.extend([
        "",
        "## Working Set",
    ])
    for item in normalized["working_set"]:
        lines.append(f"- {item}")
    if not normalized["working_set"]:
        lines.append("- (none)")
    lines.extend(["", "## Open Loops"])
    for item in normalized["open_loops"]:
        lines.append(f"- {item}")
    if not normalized["open_loops"]:
        lines.append("- (none)")
    lines.extend(["", "## Last Action"])
    last_action = normalized.get("last_action")
    if isinstance(last_action, dict):
        lines.append(f"- Tool: {last_action.get('tool') or '(none)'}")
        lines.append(f"- Summary: {last_action.get('summary') or '(none)'}")
        lines.append(f"- Timestamp: {last_action.get('ts') or '(none)'}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def _infer_last_action(
    *,
    ability_state: dict[str, Any] | None,
    task_state: dict[str, Any] | None,
    planning_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    selected = _last_ability_step(ability_state)
    if selected is None:
        selected = _last_task_tool_step(task_state)
    if not isinstance(selected, dict):
        return None
    tool = _sanitize_line(str(selected.get("tool") or ""), max_len=40)
    if not tool:
        return None
    summary = _last_action_summary(
        tool=tool,
        step=selected,
        task_state=task_state,
        planning_context=planning_context,
    )
    if not summary:
        summary = f"Executed {tool}."
    return {
        "tool": tool,
        "summary": summary,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _last_ability_step(ability_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ability_state, dict):
        return None
    steps = ability_state.get("steps") if isinstance(ability_state.get("steps"), list) else []
    selected: dict[str, Any] | None = None
    for item in steps:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip().lower()
        if status in {"executed", "waiting_user", "failed"}:
            selected = item
    return selected


def _last_task_tool_step(task_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(task_state, dict):
        return None
    plan = task_state.get("plan") if isinstance(task_state.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    selected: dict[str, Any] | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        proposal = step.get("proposal") if isinstance(step.get("proposal"), dict) else {}
        kind = str(proposal.get("kind") or "").strip().lower()
        if kind != "call_tool":
            continue
        status = str(step.get("status") or "").strip().lower()
        if status not in {"executed", "failed"}:
            continue
        tool = _sanitize_line(str(proposal.get("tool_name") or ""), max_len=40)
        if not tool:
            continue
        selected = {"tool": tool, "status": status, "step_id": str(step.get("step_id") or "").strip()}
    return selected


def _last_action_summary(
    *,
    tool: str,
    step: dict[str, Any],
    task_state: dict[str, Any] | None,
    planning_context: dict[str, Any] | None,
) -> str:
    step_idx = step.get("idx")
    facts = []
    if isinstance(planning_context, dict):
        facts_container = planning_context.get("facts") if isinstance(planning_context.get("facts"), dict) else {}
        facts = facts_container.get("tool_results") if isinstance(facts_container.get("tool_results"), list) else []
    fact = None
    for item in facts:
        if not isinstance(item, dict):
            continue
        if step_idx is not None and item.get("step") == step_idx:
            fact = item
    if tool == "getTime":
        if isinstance(fact, dict) and fact.get("time"):
            return "Fetched current time."
        return "Fetched current time."
    if tool == "createReminder":
        return "Created a reminder."
    if tool == "stt_transcribe":
        return "Transcribed an audio asset."
    if tool == "python_subprocess":
        return "Ran a subprocess command."
    if tool == "askQuestion":
        return "Asked the user for clarification."
    if isinstance(task_state, dict):
        facts = task_state.get("facts") if isinstance(task_state.get("facts"), dict) else {}
        step_id = str(step.get("step_id") or "").strip()
        if step_id and isinstance(facts.get(step_id), dict):
            result = facts[step_id].get("result")
            if tool == "local_audio_output.speak":
                return "Played local audio output."
            if tool == "getTime":
                return "Fetched current time."
            if tool == "createReminder":
                return "Created a reminder."
    status = str(step.get("status") or "").strip().lower()
    if status == "waiting_user":
        return f"Waiting on user after {tool}."
    if status == "failed":
        return f"{tool} failed."
    return f"Executed {tool}."


def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    session_id = str(state.get("session_id") or "").strip()
    user_id = _canonical_user_id(str(state.get("user_id") or ""))
    local_date = str(state.get("date") or "").strip()
    if not local_date:
        local_date = datetime.now(timezone.utc).date().isoformat()
    if not session_id:
        session_id = session_id_for_user_day(user_id, local_date)
    elif "|" not in session_id:
        session_id = session_id_for_user_day(user_id, local_date)
    last_action = state.get("last_action")
    action_payload: dict[str, Any] | None = None
    if isinstance(last_action, dict):
        tool = _sanitize_line(str(last_action.get("tool") or ""), max_len=40)
        summary = _sanitize_line(str(last_action.get("summary") or ""), max_len=100)
        ts = _sanitize_line(str(last_action.get("ts") or ""), max_len=40)
        if tool or summary or ts:
            action_payload = {
                "tool": tool or None,
                "summary": summary or None,
                "ts": ts or None,
            }
    return {
        "session_id": session_id,
        "user_id": user_id,
        "date": local_date,
        "rev": max(int(state.get("rev") or 0), 0),
        "channels_seen": _merge_channels(state.get("channels_seen"), None),
        "recent_conversation": _normalize_recent_conversation(state.get("recent_conversation")),
        "working_set": _normalize_list(state.get("working_set"), limit=_MAX_WORKING_SET),
        "open_loops": _normalize_list(state.get("open_loops"), limit=_MAX_OPEN_LOOPS),
        "last_action": action_payload,
    }


def _merge_channels(existing: Any, new_channel: str | None) -> list[str]:
    channels: list[str] = []
    raw = existing if isinstance(existing, list) else []
    for item in raw:
        candidate = _sanitize_line(str(item or "").strip().lower(), max_len=20)
        if candidate and candidate not in channels:
            channels.append(candidate)
    if new_channel:
        normalized = _sanitize_line(str(new_channel).strip().lower(), max_len=20)
        if normalized and normalized not in channels:
            channels.append(normalized)
    return channels[:_MAX_CHANNELS]


def _normalize_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        line = _sanitize_line(str(item or ""))
        if not line or line in result:
            continue
        result.append(line)
        if len(result) >= limit:
            break
    return result


def _normalize_recent_conversation(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        user_text = _sanitize_line(str(item.get("user") or ""), max_len=120)
        assistant_text = _sanitize_line(str(item.get("assistant") or ""), max_len=120)
        if not user_text and not assistant_text:
            continue
        normalized.append({"user": user_text, "assistant": assistant_text})
        if len(normalized) >= _MAX_RECENT_TURNS:
            normalized = normalized[-_MAX_RECENT_TURNS:]
    return normalized


def _append_recent_turn(*, existing: Any, user_text: str, assistant_text: str) -> list[dict[str, str]]:
    turns = _normalize_recent_conversation(existing)
    if not user_text and not assistant_text:
        return turns
    turns.append({"user": _sanitize_line(user_text, max_len=120), "assistant": _sanitize_line(assistant_text, max_len=120)})
    return turns[-_MAX_RECENT_TURNS:]


def _sanitize_line(value: str, *, max_len: int = _MAX_LINE_LENGTH) -> str:
    rendered = str(value or "").replace("\n", " ").replace("\r", " ")
    rendered = " ".join(rendered.split())
    rendered = _NAMED_SECRET_PATTERN.sub(lambda m: f"{m.group(1)}=<redacted>", rendered)
    rendered = _TOKEN_SECRET_PATTERN.sub("<redacted>", rendered)
    for pattern in _PATH_PATTERNS:
        rendered = pattern.sub("[path]", rendered)
    rendered = rendered.strip()
    if len(rendered) > max_len:
        return f"{rendered[:max_len-3]}..."
    return rendered


def _canonical_user_id(user_id: str) -> str:
    candidate = str(user_id or "").strip()
    if not candidate:
        return "anonymous"
    candidate = candidate.replace("|", "_")
    candidate = candidate.replace("\n", "_")
    return candidate


def _session_json_path(*, user_id: str, local_date: str, root_dir: Path | None) -> Path:
    safe_user = _safe_path_segment(user_id)
    base = _sessions_root(root_dir=root_dir)
    return base / "users" / safe_user / f"{local_date}.state.json"


def _session_markdown_path(*, user_id: str, local_date: str, root_dir: Path | None) -> Path:
    safe_user = _safe_path_segment(user_id)
    base = _sessions_root(root_dir=root_dir)
    return base / "users" / safe_user / f"{local_date}.md"


def _sessions_root(*, root_dir: Path | None) -> Path:
    if isinstance(root_dir, Path):
        return root_dir
    return Path(__file__).resolve().parents[3] / "sessions"


def _safe_path_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "anonymous"


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
