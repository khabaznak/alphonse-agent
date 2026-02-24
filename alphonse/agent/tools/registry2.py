from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from alphonse.agent.tools.spec import SafetyLevel, ToolSpec


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass
class ToolRegistry:
    _specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self._specs[str(spec.key)] = spec

    def get(self, key: str) -> ToolSpec | None:
        return self._specs.get(str(key))

    def specs(self) -> list[ToolSpec]:
        return sorted(self._specs.values(), key=lambda item: item.key)

    def specs_for_catalog(self) -> list[ToolSpec]:
        return [spec for spec in self.specs() if spec.expose_in_catalog]

    def specs_for_schemas(self) -> list[ToolSpec]:
        return [spec for spec in self.specs() if spec.expose_in_schemas]


def build_planner_tool_registry(extra_specs: Iterable[ToolSpec] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in _default_specs():
        registry.register(spec)
    for spec in extra_specs or []:
        registry.register(spec)
    return registry


def planner_tool_schemas_from_specs(registry: ToolRegistry) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for spec in registry.specs_for_schemas():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.key,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
        )
    return schemas


def _default_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            key="askQuestion",
            description="Ask the user one clear question and wait for their answer.",
            when_to_use="Only when required user data is missing.",
            returns="user_answer_captured",
            input_schema=_object_schema(
                properties={"question": {"type": "string"}},
                required=["question"],
            ),
            domain_tags=["planning", "clarification"],
            safety_level=SafetyLevel.LOW,
            examples=[{"question": "What exact time should I use for this reminder?"}],
        ),
        ToolSpec(
            key="getTime",
            description="Get your current time now.",
            when_to_use="Use for current time/date and as a reference for scheduling or deadline calculations.",
            returns="current_time",
            input_schema=_object_schema(properties={}, required=[]),
            domain_tags=["time", "planning"],
            safety_level=SafetyLevel.LOW,
            examples=[{}],
        ),
        ToolSpec(
            key="createReminder",
            description="Create a reminder for someone at a specific time.",
            when_to_use="Use when the user asks to be reminded.",
            returns="scheduled_reminder_id",
            input_schema=_object_schema(
                properties={
                    "ForWhom": {"type": "string"},
                    "Time": {"type": "string"},
                    "Message": {"type": "string"},
                },
                required=["ForWhom", "Time", "Message"],
            ),
            domain_tags=["time", "reminders"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"ForWhom": "me", "Time": "tomorrow 8am", "Message": "take medicine"}],
        ),
        ToolSpec(
            key="sendMessage",
            description="Send a message to a recipient through a communication channel (for example Telegram).",
            when_to_use="Use when the user asks Alphonse to deliver a direct message to someone.",
            returns="delivery_status",
            input_schema=_object_schema(
                properties={
                    "To": {"type": "string"},
                    "Message": {"type": "string"},
                    "Channel": {"type": "string"},
                    "Urgency": {"type": "string"},
                },
                required=["To", "Message"],
            ),
            domain_tags=["communication", "messaging", "delivery"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"To": "Gabriela", "Message": "Hola Gaby, Alex llegará para cenar.", "Channel": "telegram"}],
        ),
        ToolSpec(
            key="local_audio_output.speak",
            description="Speak text out loud on the local computer using OS-native TTS.",
            when_to_use="Use for local spoken output when requested.",
            returns="local_audio_output_status",
            input_schema=_object_schema(
                properties={
                    "text": {"type": "string"},
                    "voice": {"type": "string"},
                    "blocking": {"type": "boolean"},
                    "volume": {"type": "number"},
                },
                required=["text"],
            ),
            domain_tags=["audio", "output"],
            safety_level=SafetyLevel.LOW,
            examples=[{"text": "Hola, te escucho.", "voice": "Jorge"}],
        ),
        ToolSpec(
            key="stt_transcribe",
            description="Transcribe an audio asset by asset_id into text.",
            when_to_use="Use when the incoming message includes an audio asset and you need its transcript.",
            returns="transcript text and optional segments",
            input_schema=_object_schema(
                properties={
                    "asset_id": {"type": "string"},
                    "language_hint": {"type": "string"},
                },
                required=["asset_id"],
            ),
            domain_tags=["audio", "transcription"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"asset_id": "asset_123", "language_hint": "es-MX"}],
        ),
        ToolSpec(
            key="telegramGetFileMeta",
            description="Resolve Telegram file metadata from a file_id.",
            when_to_use="Use when telegram file metadata is required before download/transcription.",
            returns="telegram_file_meta",
            input_schema=_object_schema(
                properties={"file_id": {"type": "string"}},
                required=["file_id"],
            ),
            domain_tags=["telegram", "files"],
            safety_level=SafetyLevel.MEDIUM,
            expose_in_catalog=False,
            examples=[{"file_id": "AgACAgQAAx..."}],
        ),
        ToolSpec(
            key="telegramDownloadFile",
            description="Download a Telegram file by file_id and return local path details.",
            when_to_use="Use when a telegram file must be downloaded for downstream processing.",
            returns="download_path_details",
            input_schema=_object_schema(
                properties={
                    "file_id": {"type": "string"},
                    "sandbox_alias": {"type": "string"},
                    "relative_path": {"type": "string"},
                },
                required=["file_id"],
            ),
            domain_tags=["telegram", "files"],
            safety_level=SafetyLevel.HIGH,
            expose_in_catalog=False,
            examples=[{"file_id": "AgACAgQAAx...", "sandbox_alias": "telegram"}],
        ),
        ToolSpec(
            key="transcribeTelegramAudio",
            description="Download Telegram audio by file_id and transcribe it to text.",
            when_to_use="Use when an inbound telegram audio message should be transcribed directly.",
            returns="transcript text",
            input_schema=_object_schema(
                properties={
                    "file_id": {"type": "string"},
                    "language": {"type": "string"},
                    "sandbox_alias": {"type": "string"},
                },
                required=["file_id"],
            ),
            domain_tags=["telegram", "audio", "transcription"],
            safety_level=SafetyLevel.HIGH,
            expose_in_catalog=False,
            examples=[{"file_id": "CQACAgQAAx...", "language": "es"}],
        ),
        ToolSpec(
            key="analyzeTelegramImage",
            description="Download Telegram image by file_id and analyze it with a prompt.",
            when_to_use="Use when an inbound telegram image requires semantic analysis.",
            returns="image_analysis",
            input_schema=_object_schema(
                properties={
                    "file_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "sandbox_alias": {"type": "string"},
                },
                required=[],
            ),
            domain_tags=["telegram", "image", "analysis"],
            safety_level=SafetyLevel.HIGH,
            expose_in_catalog=False,
            examples=[{"file_id": "AgACAgQAAx...", "prompt": "Describe the image briefly."}],
        ),
        ToolSpec(
            key="vision_analyze_image",
            description="Analyze a sandboxed image using Alphonse's dedicated vision model.",
            when_to_use="Use for image interpretation tasks like receipts, notes, package checks, and object descriptions.",
            returns="image_analysis",
            input_schema=_object_schema(
                properties={
                    "sandbox_alias": {"type": "string"},
                    "relative_path": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                required=["sandbox_alias", "relative_path"],
            ),
            domain_tags=["vision", "image", "analysis"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[
                {
                    "sandbox_alias": "telegram_files",
                    "relative_path": "users/8553589429/images/abc123.bin",
                    "prompt": "Extract the items and totals from this receipt.",
                }
            ],
        ),
        ToolSpec(
            key="getMySettings",
            description="Get runtime settings for current conversation context.",
            when_to_use="Use before time or language-sensitive decisions when settings are needed.",
            returns="settings",
            input_schema=_object_schema(properties={}, required=[]),
            domain_tags=["context", "settings"],
            safety_level=SafetyLevel.LOW,
            examples=[{}],
        ),
        ToolSpec(
            key="getUserDetails",
            description="Get known user and channel details for current conversation context.",
            when_to_use="Use when user identity/context details are needed before planning or scheduling.",
            returns="user_details",
            input_schema=_object_schema(properties={}, required=[]),
            domain_tags=["context", "identity"],
            safety_level=SafetyLevel.LOW,
            examples=[{}],
        ),
        ToolSpec(
            key="scratchpad_create",
            description="Create a new scratchpad document and return its doc_id handle.",
            when_to_use="Use when you need to start a durable planning note, chore list, or running log.",
            returns="doc metadata with doc_id",
            input_schema=_object_schema(
                properties={
                    "title": {"type": "string"},
                    "scope": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "template": {"type": ["string", "null"]},
                },
                required=["title"],
            ),
            domain_tags=["productivity", "caregiving", "memory"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"title": "Weekly chores plan", "scope": "household", "tags": ["chores", "family"]}],
        ),
        ToolSpec(
            key="scratchpad_append",
            description="Append a timestamped block to an existing scratchpad document.",
            when_to_use="Use when adding progress updates or notes without rewriting existing history.",
            returns="append confirmation",
            input_schema=_object_schema(
                properties={
                    "doc_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                required=["doc_id", "text"],
            ),
            domain_tags=["productivity", "caregiving", "memory"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"doc_id": "sp_1234abcd", "text": "Buy milk and eggs tonight."}],
        ),
        ToolSpec(
            key="scratchpad_read",
            description="Read a scratchpad document by doc_id with bounded modes for token safety.",
            when_to_use="Use to inspect notes before deciding next action, especially with mode='tail' or 'summary'.",
            returns="document content snapshot",
            input_schema=_object_schema(
                properties={
                    "doc_id": {"type": "string"},
                    "mode": {"type": "string", "enum": ["full", "summary", "tail", "head"]},
                    "max_chars": {"type": "integer"},
                },
                required=["doc_id"],
            ),
            domain_tags=["productivity", "caregiving", "memory"],
            safety_level=SafetyLevel.LOW,
            examples=[{"doc_id": "sp_1234abcd", "mode": "tail", "max_chars": 3000}],
        ),
        ToolSpec(
            key="scratchpad_list",
            description="List scratchpad documents filtered by scope or tag.",
            when_to_use="Use to find candidate documents before reading/appending.",
            returns="list of scratchpad docs",
            input_schema=_object_schema(
                properties={
                    "scope": {"type": ["string", "null"]},
                    "tag": {"type": ["string", "null"]},
                    "limit": {"type": "integer"},
                },
                required=[],
            ),
            domain_tags=["productivity", "caregiving", "memory"],
            safety_level=SafetyLevel.LOW,
            examples=[{"scope": "household", "limit": 10}],
        ),
        ToolSpec(
            key="scratchpad_search",
            description="Search scratchpad documents by query with ranking over title, tags, and content.",
            when_to_use="Use when you need to locate prior notes quickly by keyword.",
            returns="ranked search hits",
            input_schema=_object_schema(
                properties={
                    "query": {"type": "string"},
                    "scope": {"type": ["string", "null"]},
                    "tags_any": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                },
                required=["query"],
            ),
            domain_tags=["productivity", "caregiving", "memory"],
            safety_level=SafetyLevel.LOW,
            examples=[{"query": "chores", "scope": "household", "limit": 5}],
        ),
        ToolSpec(
            key="scratchpad_fork",
            description="Create a new scratchpad document forked from an existing document.",
            when_to_use="Use when a substantial rewrite is needed while preserving immutable append history.",
            returns="fork metadata",
            input_schema=_object_schema(
                properties={
                    "doc_id": {"type": "string"},
                    "new_title": {"type": ["string", "null"]},
                },
                required=["doc_id"],
            ),
            domain_tags=["productivity", "caregiving", "memory"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"doc_id": "sp_1234abcd", "new_title": "Weekly chores plan v2"}],
        ),
        ToolSpec(
            key="job_create",
            description="Create a scheduled job with RRULE timing, payload routing, safety, and retry policy.",
            when_to_use="Use when the user asks for recurring automations or scheduled background actions.",
            returns="job_id and next_run_at",
            input_schema=_object_schema(
                properties={
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "schedule": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "dtstart": {"type": "string"},
                            "rrule": {"type": "string"},
                        },
                        "required": ["type", "dtstart", "rrule"],
                        "additionalProperties": False,
                    },
                    "payload_type": {"type": "string", "enum": ["job_ability", "tool_call", "prompt_to_brain", "internal_event"]},
                    "payload": {"type": "object"},
                    "timezone": {"type": "string"},
                    "domain_tags": {"type": "array", "items": {"type": "string"}},
                    "safety_level": {"type": "string"},
                    "requires_confirmation": {"type": "boolean"},
                    "retry_policy": {"type": "object"},
                    "idempotency": {"type": "object"},
                    "enabled": {"type": "boolean"},
                },
                required=["name", "description", "schedule", "payload_type", "payload"],
            ),
            domain_tags=["automation", "jobs", "productivity"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[
                {
                    "name": "Weekly chores digest",
                    "description": "Send Sunday chores reminder",
                    "schedule": {
                        "type": "rrule",
                        "dtstart": "2026-02-17T09:00:00-06:00",
                        "rrule": "FREQ=WEEKLY;BYDAY=SU;BYHOUR=18;BYMINUTE=0",
                    },
                    "payload_type": "prompt_to_brain",
                    "payload": {"prompt_text": "Prepare weekly chores digest for the family."},
                }
            ],
        ),
        ToolSpec(
            key="job_list",
            description="List scheduled jobs with filtering by enabled state and domain tag.",
            when_to_use="Use to review configured jobs before editing/running/deleting.",
            returns="jobs summary list",
            input_schema=_object_schema(
                properties={
                    "enabled": {"type": ["boolean", "null"]},
                    "domain_tag": {"type": ["string", "null"]},
                    "limit": {"type": "integer"},
                },
                required=[],
            ),
            domain_tags=["automation", "jobs", "productivity"],
            safety_level=SafetyLevel.LOW,
            examples=[{"enabled": True, "limit": 20}],
        ),
        ToolSpec(
            key="job_pause",
            description="Pause a scheduled job so it no longer auto-triggers.",
            when_to_use="Use when the user wants to temporarily disable an automation.",
            returns="job enabled state and next_run_at",
            input_schema=_object_schema(
                properties={"job_id": {"type": "string"}},
                required=["job_id"],
            ),
            domain_tags=["automation", "jobs", "control"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"job_id": "job_abc123"}],
        ),
        ToolSpec(
            key="job_resume",
            description="Resume a paused job and recompute next run time.",
            when_to_use="Use when the user wants an automation active again.",
            returns="job enabled state and next_run_at",
            input_schema=_object_schema(
                properties={"job_id": {"type": "string"}},
                required=["job_id"],
            ),
            domain_tags=["automation", "jobs", "control"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"job_id": "job_abc123"}],
        ),
        ToolSpec(
            key="job_delete",
            description="Delete a job definition permanently.",
            when_to_use="Use when the user asks to remove an automation.",
            returns="deletion status",
            input_schema=_object_schema(
                properties={"job_id": {"type": "string"}},
                required=["job_id"],
            ),
            domain_tags=["automation", "jobs", "control"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"job_id": "job_abc123"}],
        ),
        ToolSpec(
            key="job_run_now",
            description="Trigger immediate execution of a job regardless of schedule.",
            when_to_use="Use to test or manually force a scheduled job.",
            returns="execution id and status",
            input_schema=_object_schema(
                properties={
                    "job_id": {"type": "string"},
                    "job_name": {"type": "string"},
                    "name": {"type": "string"},
                },
                required=[],
            ),
            domain_tags=["automation", "jobs", "control"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"job_id": "job_abc123"}, {"job_name": "Weekly chores digest"}],
        ),
        ToolSpec(
            key="user_register_from_contact",
            description="Register or update a user from a shared Telegram contact with strict admin authorization.",
            when_to_use="Use when an admin asks to add/register a person and shares their contact.",
            returns="registered user metadata and proactive intro signal id",
            input_schema=_object_schema(
                properties={
                    "display_name": {"type": ["string", "null"]},
                    "role": {"type": ["string", "null"]},
                    "relationship": {"type": ["string", "null"]},
                    "contact_user_id": {"type": ["string", "integer", "null"]},
                    "contact_first_name": {"type": ["string", "null"]},
                    "contact_last_name": {"type": ["string", "null"]},
                    "contact_phone": {"type": ["string", "null"]},
                },
                required=[],
            ),
            domain_tags=["identity", "admin", "onboarding"],
            safety_level=SafetyLevel.MEDIUM,
            requires_confirmation=False,
            examples=[{"display_name": "Maria Perez", "role": "family", "relationship": "sister"}],
        ),
        ToolSpec(
            key="user_remove_from_contact",
            description="Deactivate a registered user from a shared Telegram contact with strict admin authorization.",
            when_to_use="Use when an admin asks to remove a person and shares their contact.",
            returns="deactivation status",
            input_schema=_object_schema(
                properties={
                    "contact_user_id": {"type": ["string", "integer", "null"]},
                },
                required=[],
            ),
            domain_tags=["identity", "admin", "access-control"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"contact_user_id": "8553589429"}],
        ),
        ToolSpec(
            key="user_search",
            description="Search registered users by partial display name and include channel resolver identifiers.",
            when_to_use="Use before sending messages when recipient identity is uncertain and needs read-only lookup.",
            returns="matching users",
            input_schema=_object_schema(
                properties={
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "active_only": {"type": "boolean"},
                },
                required=["query"],
            ),
            domain_tags=["identity", "users", "lookup"],
            safety_level=SafetyLevel.LOW,
            examples=[{"query": "Gab", "limit": 5, "active_only": True}],
        ),
        ToolSpec(
            key="terminal_execute",
            description="Execute terminal commands under global Alphonse execution mode and sandbox policy.",
            when_to_use="Use for constrained terminal-like operations when explicit tools are insufficient.",
            returns="command stdout/stderr/exit_code with policy metadata",
            input_schema=_object_schema(
                properties={
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                },
                required=["command"],
            ),
            domain_tags=["ops", "terminal", "automation"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"command": "ls -la", "cwd": ".", "timeout_seconds": 20}],
        ),
        ToolSpec(
            key="terminal_command_submit",
            description="Submit a terminal command for asynchronous execution and return a command_id for polling.",
            when_to_use="Use for long-running commands where blocking the current turn is undesirable.",
            returns="command_id and initial status",
            input_schema=_object_schema(
                properties={
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                    "sandbox_alias": {"type": "string"},
                },
                required=["command"],
            ),
            domain_tags=["ops", "terminal", "automation"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"command": "ollama pull llama3.2", "timeout_seconds": 1200, "sandbox_alias": "main"}],
        ),
        ToolSpec(
            key="terminal_command_status",
            description="Get status and output for an asynchronous terminal command by command_id.",
            when_to_use="Use after terminal_command_submit to monitor progress and retrieve stdout/stderr when complete.",
            returns="command status, done flag, and output",
            input_schema=_object_schema(
                properties={"command_id": {"type": "string"}},
                required=["command_id"],
            ),
            domain_tags=["ops", "terminal", "automation"],
            safety_level=SafetyLevel.LOW,
            examples=[{"command_id": "6e9027bf-f81d-4277-bc57-8c40a7cb389f"}],
        ),
        ToolSpec(
            key="ssh_terminal",
            description="Execute a command on a remote SSH host using Paramiko.",
            when_to_use="Use for explicit remote SSH operations that require host/user credentials and command execution.",
            returns="remote command stdout/stderr/exit_code",
            input_schema=_object_schema(
                properties={
                    "host": {"type": "string"},
                    "username": {"type": "string"},
                    "command": {"type": "string"},
                    "port": {"type": "integer"},
                    "password": {"type": "string"},
                    "private_key_path": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                    "connect_timeout_seconds": {"type": "number"},
                },
                required=["host", "username", "command"],
            ),
            domain_tags=["ops", "ssh", "remote"],
            safety_level=SafetyLevel.CRITICAL,
            requires_confirmation=True,
            examples=[
                {
                    "host": "192.168.1.20",
                    "username": "pi",
                    "command": "uname -a",
                    "timeout_seconds": 30,
                }
            ],
        ),
        ToolSpec(
            key="python_subprocess",
            description=(
                "Execute a Python subprocess command on the local system. Use to install missing tools or for other "
                "system-level operations. Be cautious with this tool and avoid running untrusted commands."
            ),
            when_to_use="Use only for admin-authorized maintenance tasks when built-in tools are insufficient.",
            returns="command output, error message, and exit code",
            input_schema=_object_schema(
                properties={
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                },
                required=["command"],
            ),
            domain_tags=["ops", "maintenance", "system"],
            safety_level=SafetyLevel.CRITICAL,
            requires_confirmation=True,
            examples=[{"command": "which python3", "timeout_seconds": 20}],
        ),
    ]
