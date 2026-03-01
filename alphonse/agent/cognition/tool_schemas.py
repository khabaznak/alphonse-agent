from __future__ import annotations

from typing import Any


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_TOOL_SCHEMA_DEFS: dict[str, dict[str, Any]] = {
    "create_reminder": {
        "description": "Create a reminder for someone at a specific time.",
        "parameters": _object_schema(
            {"ForWhom": {"type": "string"}, "Time": {"type": "string"}, "Message": {"type": "string"}},
            ["ForWhom", "Time", "Message"],
        ),
    },
    "domotics.query": {
        "description": "Query domotics states through the configured backend (Home Assistant in v1).",
        "parameters": _object_schema(
            {
                "kind": {"type": "string", "enum": ["states", "state"]},
                "entity_id": {"type": "string"},
                "filters": {"type": "object"},
            },
            ["kind"],
        ),
    },
    "domotics.execute": {
        "description": "Execute a domotics service action and optionally verify effect via readback.",
        "parameters": _object_schema(
            {
                "domain": {"type": "string"},
                "service": {"type": "string"},
                "data": {"type": "object"},
                "target": {"type": "object"},
                "readback": {"type": "boolean"},
                "readback_entity_id": {"type": "string"},
                "expected_state": {"type": "string"},
                "expected_attributes": {"type": "object"},
            },
            ["domain", "service"],
        ),
    },
    "domotics.subscribe": {
        "description": "Subscribe to domotics events for a short capture window and return normalized events.",
        "parameters": _object_schema(
            {
                "event_type": {"type": "string"},
                "duration_seconds": {"type": "number"},
                "filters": {"type": "object"},
                "max_events": {"type": "integer"},
            },
            [],
        ),
    },
    "get_my_settings": {
        "description": "Get runtime settings for current conversation context.",
        "parameters": _object_schema({}, []),
    },
    "get_time": {"description": "Get your current time now.", "parameters": _object_schema({}, [])},
    "get_user_details": {
        "description": "Get known user and channel details for current conversation context.",
        "parameters": _object_schema({}, []),
    },
    "search_episodes": {
        "description": "Search episodic memory entries for the current user with optional mission and time filters.",
        "parameters": _object_schema(
            {
                "query": {"type": "string"},
                "user_id": {"type": "string"},
                "mission_id": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "limit": {"type": "integer"},
            },
            ["query"],
        ),
    },
    "get_mission": {
        "description": "Get mission details from memory by mission_id.",
        "parameters": _object_schema(
            {
                "mission_id": {"type": "string"},
                "user_id": {"type": "string"},
            },
            ["mission_id"],
        ),
    },
    "list_active_missions": {
        "description": "List active missions from memory for the current user.",
        "parameters": _object_schema({"user_id": {"type": "string"}}, []),
    },
    "get_workspace_pointer": {
        "description": "Get a workspace pointer by key from memory for the current user.",
        "parameters": _object_schema(
            {
                "key": {"type": "string"},
                "user_id": {"type": "string"},
            },
            ["key"],
        ),
    },
    "job_create": {
        "description": "Create a scheduled job with RRULE timing, payload routing, safety, and retry policy.",
        "parameters": _object_schema(
            {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "schedule": {"type": "object"},
                "payload_type": {"type": "string", "enum": ["job_ability", "tool_call", "prompt_to_brain", "internal_event"]},
                "payload": {"type": "object"},
                "enabled": {"type": "boolean"},
                "domain_tag": {"type": "string"},
            },
            ["name", "description", "schedule", "payload_type", "payload"],
        ),
    },
    "job_delete": {
        "description": "Delete a job definition permanently.",
        "parameters": _object_schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    "job_list": {
        "description": "List scheduled jobs with filtering by enabled state and domain tag.",
        "parameters": _object_schema({"enabled": {"type": "boolean"}, "domain_tag": {"type": "string"}, "limit": {"type": "integer"}}, []),
    },
    "job_pause": {"description": "Pause a scheduled job so it no longer auto-triggers.", "parameters": _object_schema({"job_id": {"type": "string"}}, ["job_id"])},
    "job_resume": {
        "description": "Resume a paused job and recompute next run time.",
        "parameters": _object_schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    "job_run_now": {
        "description": "Trigger immediate execution of a job regardless of schedule.",
        "parameters": _object_schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    "local_audio_output_render": {
        "description": "Render text to an audio file on the local machine for downstream delivery integrations.",
        "parameters": _object_schema(
            {
                "text": {"type": "string"},
                "voice": {"type": "string"},
                "output_dir": {"type": "string"},
                "filename_prefix": {"type": "string"},
                "format": {"type": "string", "enum": ["aiff", "m4a", "ogg"]},
            },
            ["text"],
        ),
    },
    "local_audio_output_speak": {
        "description": "Speak text out loud on the local computer using OS-native TTS.",
        "parameters": _object_schema(
            {"text": {"type": "string"}, "voice": {"type": "string"}, "blocking": {"type": "boolean"}, "volume": {"type": "number"}},
            ["text"],
        ),
    },
    "mcp_call": {
        "description": "Execute a named operation via an MCP profile through a controlled connector and policy envelope.",
        "parameters": _object_schema(
            {
                "profile": {"type": "string"},
                "operation": {"type": "string"},
                "arguments": {"type": "object"},
                "headless": {"type": "boolean"},
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "number"},
            },
            ["profile", "operation"],
        ),
    },
    "send_message": {
        "description": "Send a message to a recipient through a communication channel (for example Telegram).",
        "parameters": _object_schema(
            {
                "To": {"type": "string"},
                "Message": {"type": "string"},
                "Channel": {"type": "string"},
                "Urgency": {"type": "string"},
                "DeliveryMode": {"type": "string", "enum": ["text", "audio"]},
                "AudioFilePath": {"type": "string"},
                "AsVoice": {"type": "boolean"},
                "Caption": {"type": "string"},
            },
            ["To", "Message"],
        ),
    },
    "send_voice_note": {
        "description": "Send a Telegram-style voice note to a recipient.",
        "parameters": _object_schema(
            {
                "To": {"type": "string"},
                "AudioFilePath": {"type": "string"},
                "Caption": {"type": "string"},
                "Message": {"type": "string"},
                "Channel": {"type": "string"},
                "Urgency": {"type": "string"},
                "AsVoice": {"type": "boolean"},
            },
            ["To", "AudioFilePath"],
        ),
    },
    "ssh_terminal": {
        "description": "Execute a command on a remote SSH host using Paramiko.",
        "parameters": _object_schema(
            {
                "host": {"type": "string"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "key_path": {"type": "string"},
                "command": {"type": "string"},
                "port": {"type": "integer"},
                "timeout_seconds": {"type": "number"},
            },
            ["host", "command"],
        ),
    },
    "stt_transcribe": {
        "description": "Transcribe an audio asset by asset_id into text.",
        "parameters": _object_schema({"asset_id": {"type": "string"}, "language_hint": {"type": "string"}}, ["asset_id"]),
    },
    "telegram_download_file": {
        "description": "Download a Telegram file by file_id and return local path details.",
        "parameters": _object_schema(
            {"file_id": {"type": "string"}, "sandbox_alias": {"type": "string"}, "relative_path": {"type": "string"}},
            ["file_id"],
        ),
    },
    "telegram_get_file_meta": {
        "description": "Resolve Telegram file metadata from a file_id.",
        "parameters": _object_schema({"file_id": {"type": "string"}}, ["file_id"]),
    },
    "terminal_async": {
        "description": "Submit a terminal command for asynchronous execution and return a command_id for polling.",
        "parameters": _object_schema(
            {
                "command": {"type": "string"},
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "number"},
                "sandbox_alias": {"type": "string"},
            },
            ["command"],
        ),
    },
    "terminal_async_command_status": {
        "description": "Get status and output for an asynchronous terminal command by command_id.",
        "parameters": _object_schema({"command_id": {"type": "string"}}, ["command_id"]),
    },
    "terminal_sync": {
        "description": "Execute terminal commands under global Alphonse execution mode and sandbox policy.",
        "parameters": _object_schema({"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout_seconds": {"type": "number"}}, ["command"]),
    },
    "transcribe_telegram_audio": {
        "description": "Download Telegram audio by file_id and transcribe it to text.",
        "parameters": _object_schema({"file_id": {"type": "string"}, "language": {"type": "string"}, "sandbox_alias": {"type": "string"}}, ["file_id"]),
    },
    "user_register_from_contact": {
        "description": "Register or update a user from a shared Telegram contact with strict admin authorization.",
        "parameters": _object_schema({"contact": {"type": "object"}, "state": {"type": "object"}}, ["contact"]),
    },
    "user_remove_from_contact": {
        "description": "Deactivate a registered user from a shared Telegram contact with strict admin authorization.",
        "parameters": _object_schema({"phone_number": {"type": "string"}, "state": {"type": "object"}}, ["phone_number"]),
    },
    "user_search": {
        "description": "Search registered users by partial display name and include channel resolver identifiers.",
        "parameters": _object_schema({"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
    },
    "vision_analyze_image": {
        "description": "Analyze a sandboxed image for semantic understanding and description using the local vision model.",
        "parameters": _object_schema(
            {"sandbox_alias": {"type": "string"}, "relative_path": {"type": "string"}, "prompt": {"type": "string"}},
            ["sandbox_alias", "relative_path"],
        ),
    },
    "vision_extract": {
        "description": "Extract visible text (OCR) from a sandboxed image using the local vision model.",
        "parameters": _object_schema(
            {"sandbox_alias": {"type": "string"}, "relative_path": {"type": "string"}, "prompt": {"type": "string"}},
            ["sandbox_alias", "relative_path"],
        ),
    },
}


def canonical_tool_names(tool_registry: Any) -> list[str]:
    if not hasattr(tool_registry, "keys"):
        return []
    out: list[str] = []
    for name in tool_registry.keys():
        key = str(name or "").strip()
        if not key:
            continue
        if key not in _TOOL_SCHEMA_DEFS:
            continue
        # Canonical LLM-facing names only.
        if "." in key and not key.startswith("domotics."):
            continue
        if any(ch.isupper() for ch in key):
            continue
        out.append(key)
    return sorted(set(out))


def llm_tool_schemas(tool_registry: Any) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for name in canonical_tool_names(tool_registry):
        entry = _TOOL_SCHEMA_DEFS.get(name)
        if not isinstance(entry, dict):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(entry.get("description") or "Tool available."),
                    "parameters": dict(entry.get("parameters") or {}),
                },
            }
        )
    return tools


def tool_descriptions(tool_registry: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in canonical_tool_names(tool_registry):
        entry = _TOOL_SCHEMA_DEFS.get(name)
        if not isinstance(entry, dict):
            continue
        out[name] = str(entry.get("description") or "Tool available.")
    return out


def tool_parameters(tool_name: str) -> dict[str, Any] | None:
    entry = _TOOL_SCHEMA_DEFS.get(str(tool_name or "").strip())
    if not isinstance(entry, dict):
        return None
    params = entry.get("parameters")
    return dict(params) if isinstance(params, dict) else None


def required_args(tool_name: str) -> list[str]:
    params = tool_parameters(tool_name)
    if not isinstance(params, dict):
        return []
    required = params.get("required")
    if not isinstance(required, list):
        return []
    return [str(item) for item in required if str(item)]
