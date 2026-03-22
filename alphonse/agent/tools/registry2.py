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


def _permissive_output_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": True}


@dataclass
class ToolRegistry:
    _specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self._specs[str(spec.canonical_name)] = spec

    def get(self, key: str) -> ToolSpec | None:
        return self._specs.get(str(key))

    def specs(self) -> list[ToolSpec]:
        return sorted(self._specs.values(), key=lambda item: item.canonical_name)

    def specs_for_catalog(self) -> list[ToolSpec]:
        return [spec for spec in self.specs() if spec.expose_in_catalog]

    def specs_for_schemas(self) -> list[ToolSpec]:
        return [spec for spec in self.specs() if spec.expose_in_schemas and spec.visible_to_agent and not spec.deprecated]


def build_planner_tool_registry(extra_specs: Iterable[ToolSpec] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    disabled_keys = {"terminal_async", "terminal_async_command_status"}
    for spec in _default_specs():
        if spec.canonical_name in disabled_keys:
            continue
        registry.register(spec)
    for spec in extra_specs or []:
        if spec.canonical_name in disabled_keys:
            continue
        registry.register(spec)
    return registry


def planner_tool_schemas_from_specs(registry: ToolRegistry) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for spec in registry.specs_for_schemas():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.canonical_name,
                    "description": spec.summary,
                    "parameters": spec.input_schema,
                },
            }
        )
    return schemas


def _default_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            canonical_name="askQuestion",
            summary="Ask the user one clear question and wait for their answer.",
            description="Ask the user one clear question and wait for their answer.",
            when_to_use="Only when required user data is missing.",
            returns="user_answer_captured",
            input_schema=_object_schema(
                properties={"question": {"type": "string"}},
                required=["question"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["planning", "clarification"],
            safety_level=SafetyLevel.LOW,
            examples=[{"question": "What exact time should I use for this reminder?"}],
        ),
        ToolSpec(
            canonical_name="get_time",
            summary="Get your current time now.",
            description="Get your current time now.",
            when_to_use="Use for current time/date and as a reference for scheduling or deadline calculations.",
            returns="current_time",
            input_schema=_object_schema(properties={}, required=[]),
            output_schema=_permissive_output_schema(),
            domain_tags=["time", "planning"],
            aliases=["clock", "getTime"],
            safety_level=SafetyLevel.LOW,
            examples=[{}],
        ),
        ToolSpec(
            canonical_name="create_reminder",
            summary="Create a reminder for someone at a specific time.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["time", "reminders"],
            aliases=["createReminder"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"ForWhom": "me", "Time": "tomorrow 8am", "Message": "take medicine"}],
        ),
        ToolSpec(
            canonical_name="send_message",
            summary="Send a message to a recipient through a communication channel (for example Telegram).",
            description="Send a message to a recipient through a communication channel (for example Telegram).",
            when_to_use="Use when the user asks Alphonse to deliver a direct message to someone.",
            returns="delivery_status",
            input_schema=_object_schema(
                properties={
                    "To": {"type": "string"},
                    "Message": {"type": "string"},
                    "Channel": {"type": "string"},
                    "Urgency": {"type": "string"},
                    "DeliveryMode": {"type": "string", "enum": ["text", "audio"]},
                    "AudioFilePath": {"type": "string"},
                    "AsVoice": {"type": "boolean"},
                    "Caption": {"type": "string"},
                },
                required=["To", "Message"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["communication", "messaging", "delivery"],
            aliases=["sendMessage"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"To": "Gabriela", "Message": "Hola Gaby, Alex llegará para cenar.", "Channel": "telegram"}],
        ),
        ToolSpec(
            canonical_name="send_voice_note",
            summary="Send a Telegram-style voice note to a recipient.",
            description="Send a Telegram-style voice note to a recipient.",
            when_to_use="Prefer this over send_message when you must deliver a true voice note bubble in Telegram.",
            returns="delivery_status",
            input_schema=_object_schema(
                properties={
                    "To": {"type": "string"},
                    "AudioFilePath": {"type": "string"},
                    "Caption": {"type": "string"},
                    "Message": {"type": "string"},
                    "Channel": {"type": "string"},
                    "Urgency": {"type": "string"},
                    "AsVoice": {"type": "boolean"},
                },
                required=["To", "AudioFilePath"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["communication", "messaging", "delivery", "audio"],
            aliases=["sendVoiceNote"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[
                {
                    "To": "Gabriela",
                    "AudioFilePath": "/path/to/voice-note.ogg",
                    "Caption": "Te lo mando por audio.",
                    "Channel": "telegram",
                    "AsVoice": True,
                }
            ],
        ),
        ToolSpec(
            canonical_name="local_audio_output_speak",
            summary="Speak text out loud on the local computer using OS-native TTS.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["audio", "output"],
            aliases=["local_audio_output.speak"],
            safety_level=SafetyLevel.LOW,
            examples=[{"text": "Hola, te escucho.", "voice": "Jorge"}],
        ),
        ToolSpec(
            canonical_name="local_audio_output_render",
            summary="Render text to an audio file on the local machine for downstream delivery integrations.",
            description="Render text to an audio file on the local machine for downstream delivery integrations.",
            when_to_use="Use when you need a reusable audio artifact (use format='ogg' for Telegram voice notes).",
            returns="audio file path and format metadata",
            input_schema=_object_schema(
                properties={
                    "text": {"type": "string"},
                    "voice": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "filename_prefix": {"type": "string"},
                    "format": {"type": "string", "enum": ["aiff", "m4a", "ogg"]},
                },
                required=["text"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["audio", "output", "tts"],
            aliases=["local_audio_output.render"],
            safety_level=SafetyLevel.LOW,
            examples=[{"text": "Hola Alex", "format": "ogg"}],
        ),
        ToolSpec(
            canonical_name="stt_transcribe",
            summary="Transcribe an audio asset by asset_id into text.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["audio", "transcription"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"asset_id": "asset_123", "language_hint": "es-MX"}],
        ),
        ToolSpec(
            canonical_name="telegram_get_file_meta",
            summary="Resolve Telegram file metadata from a file_id.",
            description="Resolve Telegram file metadata from a file_id.",
            when_to_use="Use when telegram file metadata is required before download/transcription.",
            returns="telegram_file_meta",
            input_schema=_object_schema(
                properties={"file_id": {"type": "string"}},
                required=["file_id"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["telegram", "files"],
            aliases=["telegramGetFileMeta"],
            safety_level=SafetyLevel.MEDIUM,
            expose_in_catalog=False,
            examples=[{"file_id": "AgACAgQAAx..."}],
        ),
        ToolSpec(
            canonical_name="telegram_download_file",
            summary="Download a Telegram file by file_id and return local path details.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["telegram", "files"],
            aliases=["telegramDownloadFile"],
            safety_level=SafetyLevel.HIGH,
            expose_in_catalog=False,
            examples=[{"file_id": "AgACAgQAAx...", "sandbox_alias": "telegram"}],
        ),
        ToolSpec(
            canonical_name="transcribe_telegram_audio",
            summary="Download Telegram audio by file_id and transcribe it to text.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["telegram", "audio", "transcription"],
            aliases=["transcribeTelegramAudio"],
            safety_level=SafetyLevel.HIGH,
            expose_in_catalog=False,
            examples=[{"file_id": "CQACAgQAAx...", "language": "es"}],
        ),
        ToolSpec(
            canonical_name="vision_analyze_image",
            summary="Analyze a sandboxed image using Alphonse's dedicated local vision model.",
            description="Analyze a sandboxed image using Alphonse's dedicated local vision model.",
            when_to_use="Use for semantic image understanding tasks like scene/object description and visual interpretation.",
            returns="image_analysis",
            input_schema=_object_schema(
                properties={
                    "sandbox_alias": {"type": "string"},
                    "relative_path": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                required=["sandbox_alias", "relative_path"],
            ),
            output_schema=_permissive_output_schema(),
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
            canonical_name="vision_extract",
            summary="Extract visible text (OCR) from a sandboxed image using Alphonse's dedicated local vision model.",
            description="Extract visible text (OCR) from a sandboxed image using Alphonse's dedicated local vision model.",
            when_to_use="Use for OCR tasks over receipts, screenshots, notes, labels, and documents saved in sandbox.",
            returns="ocr_text_and_blocks",
            input_schema=_object_schema(
                properties={
                    "sandbox_alias": {"type": "string"},
                    "relative_path": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                required=["sandbox_alias", "relative_path"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["vision", "ocr", "image", "extraction"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[
                {
                    "sandbox_alias": "telegram_files",
                    "relative_path": "users/8553589429/images/abc123.bin",
                    "prompt": "Extract all text exactly as shown, preserving line breaks.",
                }
            ],
        ),
        ToolSpec(
            canonical_name="get_my_settings",
            summary="Get runtime settings for current conversation context.",
            description="Get runtime settings for current conversation context.",
            when_to_use="Use before time or language-sensitive decisions when settings are needed.",
            returns="settings",
            input_schema=_object_schema(properties={}, required=[]),
            output_schema=_permissive_output_schema(),
            domain_tags=["context", "settings"],
            safety_level=SafetyLevel.LOW,
            examples=[{}],
        ),
        ToolSpec(
            canonical_name="get_user_details",
            summary="Get known user and channel details for current conversation context.",
            description="Get known user and channel details for current conversation context.",
            when_to_use="Use when user identity/context details are needed before planning or scheduling.",
            returns="user_details",
            input_schema=_object_schema(properties={}, required=[]),
            output_schema=_permissive_output_schema(),
            domain_tags=["context", "identity"],
            safety_level=SafetyLevel.LOW,
            examples=[{}],
        ),
        ToolSpec(
            canonical_name="search_episodes",
            summary="Search episodic memory entries for the current user with optional mission and time filters.",
            description="Search episodic memory entries for the current user with optional mission and time filters.",
            when_to_use="Use when retrieving past user events/tasks by keyword or time range.",
            returns="matching episodes",
            input_schema=_object_schema(
                properties={
                    "query": {"type": "string"},
                    "user_id": {"type": "string"},
                    "mission_id": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["query"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["memory", "episodes", "search"],
            safety_level=SafetyLevel.LOW,
            examples=[{"query": "medicine", "limit": 10}],
        ),
        ToolSpec(
            canonical_name="get_mission",
            summary="Get mission details from memory by mission_id.",
            description="Get mission details from memory by mission_id.",
            when_to_use="Use when a mission identifier is known and mission context is needed.",
            returns="mission details",
            input_schema=_object_schema(
                properties={
                    "mission_id": {"type": "string"},
                    "user_id": {"type": "string"},
                },
                required=["mission_id"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["memory", "missions"],
            safety_level=SafetyLevel.LOW,
            examples=[{"mission_id": "mission_123"}],
        ),
        ToolSpec(
            canonical_name="list_active_missions",
            summary="List active missions from memory for the current user.",
            description="List active missions from memory for the current user.",
            when_to_use="Use for a quick view of currently active user missions.",
            returns="active missions list",
            input_schema=_object_schema(
                properties={"user_id": {"type": "string"}},
                required=[],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["memory", "missions"],
            safety_level=SafetyLevel.LOW,
            examples=[{}],
        ),
        ToolSpec(
            canonical_name="get_workspace_pointer",
            summary="Get a workspace pointer by key from memory for the current user.",
            description="Get a workspace pointer by key from memory for the current user.",
            when_to_use="Use when a named workspace pointer must be resolved before action.",
            returns="workspace pointer",
            input_schema=_object_schema(
                properties={
                    "key": {"type": "string"},
                    "user_id": {"type": "string"},
                },
                required=["key"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["memory", "workspace"],
            safety_level=SafetyLevel.LOW,
            examples=[{"key": "shopping_list"}],
        ),
        ToolSpec(
            canonical_name="domotics.query",
            summary="Query domotics states through the configured backend.",
            description="Query domotics states through the configured backend.",
            when_to_use="Use for read-only state inspection of home automation entities.",
            returns="domotics state payload",
            input_schema=_object_schema(
                properties={
                    "kind": {"type": "string", "enum": ["states", "state"]},
                    "entity_id": {"type": "string"},
                    "filters": {"type": "object"},
                },
                required=["kind"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["domotics", "query", "home"],
            safety_level=SafetyLevel.LOW,
            examples=[{"kind": "state", "entity_id": "light.kitchen"}],
        ),
        ToolSpec(
            canonical_name="domotics.execute",
            summary="Execute a domotics service action and optionally verify effect via readback.",
            description="Execute a domotics service action and optionally verify effect via readback.",
            when_to_use="Use for state-changing home automation operations.",
            returns="execution and optional readback result",
            input_schema=_object_schema(
                properties={
                    "domain": {"type": "string"},
                    "service": {"type": "string"},
                    "data": {"type": "object"},
                    "target": {"type": "object"},
                    "readback": {"type": "boolean"},
                    "readback_entity_id": {"type": "string"},
                    "expected_state": {"type": "string"},
                    "expected_attributes": {"type": "object"},
                },
                required=["domain", "service"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["domotics", "execute", "home"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"domain": "light", "service": "turn_on", "target": {"entity_id": "light.kitchen"}}],
        ),
        ToolSpec(
            canonical_name="domotics.subscribe",
            summary="Subscribe to domotics events for a short capture window and return normalized events.",
            description="Subscribe to domotics events for a short capture window and return normalized events.",
            when_to_use="Use when event stream observation is needed for diagnosis or confirmation.",
            returns="captured domotics events",
            input_schema=_object_schema(
                properties={
                    "event_type": {"type": "string"},
                    "duration_seconds": {"type": "number"},
                    "filters": {"type": "object"},
                    "max_events": {"type": "integer"},
                },
                required=[],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["domotics", "events", "home"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"event_type": "state_changed", "duration_seconds": 10}],
        ),
        ToolSpec(
            canonical_name="job_create",
            summary="Create a scheduled job with RRULE timing, payload routing, safety, and retry policy.",
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
            output_schema=_permissive_output_schema(),
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
            canonical_name="job_list",
            summary="List scheduled jobs with filtering by enabled state and domain tag.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["automation", "jobs", "productivity"],
            safety_level=SafetyLevel.LOW,
            examples=[{"enabled": True, "limit": 20}],
        ),
        ToolSpec(
            canonical_name="job_pause",
            summary="Pause a scheduled job so it no longer auto-triggers.",
            description="Pause a scheduled job so it no longer auto-triggers.",
            when_to_use="Use when the user wants to temporarily disable an automation.",
            returns="job enabled state and next_run_at",
            input_schema=_object_schema(
                properties={"job_id": {"type": "string"}},
                required=["job_id"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["automation", "jobs", "control"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"job_id": "job_abc123"}],
        ),
        ToolSpec(
            canonical_name="job_resume",
            summary="Resume a paused job and recompute next run time.",
            description="Resume a paused job and recompute next run time.",
            when_to_use="Use when the user wants an automation active again.",
            returns="job enabled state and next_run_at",
            input_schema=_object_schema(
                properties={"job_id": {"type": "string"}},
                required=["job_id"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["automation", "jobs", "control"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"job_id": "job_abc123"}],
        ),
        ToolSpec(
            canonical_name="job_delete",
            summary="Delete a job definition permanently.",
            description="Delete a job definition permanently.",
            when_to_use="Use when the user asks to remove an automation.",
            returns="deletion status",
            input_schema=_object_schema(
                properties={"job_id": {"type": "string"}},
                required=["job_id"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["automation", "jobs", "control"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"job_id": "job_abc123"}],
        ),
        ToolSpec(
            canonical_name="job_run_now",
            summary="Trigger immediate execution of a job regardless of schedule.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["automation", "jobs", "control"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"job_id": "job_abc123"}, {"job_name": "Weekly chores digest"}],
        ),
        ToolSpec(
            canonical_name="user_register_from_contact",
            summary="Register or update a user from a shared Telegram contact with strict admin authorization.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["identity", "admin", "onboarding"],
            safety_level=SafetyLevel.MEDIUM,
            requires_confirmation=False,
            examples=[{"display_name": "Maria Perez", "role": "family", "relationship": "sister"}],
        ),
        ToolSpec(
            canonical_name="user_remove_from_contact",
            summary="Deactivate a registered user from a shared Telegram contact with strict admin authorization.",
            description="Deactivate a registered user from a shared Telegram contact with strict admin authorization.",
            when_to_use="Use when an admin asks to remove a person and shares their contact.",
            returns="deactivation status",
            input_schema=_object_schema(
                properties={
                    "contact_user_id": {"type": ["string", "integer", "null"]},
                },
                required=[],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["identity", "admin", "access-control"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"contact_user_id": "8553589429"}],
        ),
        ToolSpec(
            canonical_name="user_search",
            summary="Search registered users by partial display name and include channel resolver identifiers.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["identity", "users", "lookup"],
            safety_level=SafetyLevel.LOW,
            examples=[{"query": "Gab", "limit": 5, "active_only": True}],
        ),
        ToolSpec(
            canonical_name="terminal_sync",
            summary="Execute terminal commands under global Alphonse execution mode and sandbox policy.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["ops", "terminal", "automation"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"command": "ls -la", "cwd": ".", "timeout_seconds": 20}],
        ),
        ToolSpec(
            canonical_name="mcp_call",
            summary="Execute a named operation via an MCP profile through a controlled connector and policy envelope.",
            description="Execute a named operation via an MCP profile through a controlled connector and policy envelope.",
            when_to_use="Use this for MCP-backed capabilities (for example Chrome MCP web search); do not call MCP binaries via terminal tools.",
            returns="mcp operation status, stdout/stderr, and policy envelope metadata",
            input_schema=_object_schema(
                properties={
                    "profile": {"type": "string"},
                    "operation": {"type": "string"},
                    "arguments": {"type": "object"},
                    "headless": {"type": "boolean"},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                },
                required=["profile", "operation"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["mcp", "automation", "integration"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[
                {
                    "profile": "chrome",
                    "operation": "web_search",
                    "arguments": {"query": "Veloswim company profile"},
                    "cwd": ".",
                }
            ],
        ),
        ToolSpec(
            canonical_name="terminal_async",
            summary="Submit a terminal command for asynchronous execution and return a command_id for polling.",
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
            output_schema=_permissive_output_schema(),
            domain_tags=["ops", "terminal", "automation"],
            safety_level=SafetyLevel.HIGH,
            requires_confirmation=True,
            examples=[{"command": "ollama pull llama3.2", "timeout_seconds": 1200, "sandbox_alias": "main"}],
        ),
        ToolSpec(
            canonical_name="terminal_async_command_status",
            summary="Get status and output for an asynchronous terminal command by command_id.",
            description="Get status and output for an asynchronous terminal command by command_id.",
            when_to_use="Use after terminal_async to monitor progress and retrieve stdout/stderr when complete.",
            returns="command status, done flag, and output",
            input_schema=_object_schema(
                properties={"command_id": {"type": "string"}},
                required=["command_id"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["ops", "terminal", "automation"],
            safety_level=SafetyLevel.LOW,
            examples=[{"command_id": "6e9027bf-f81d-4277-bc57-8c40a7cb389f"}],
        ),
        ToolSpec(
            canonical_name="ssh_terminal",
            summary="Execute a command on a remote SSH host using Paramiko.",
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
            output_schema=_permissive_output_schema(),
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
    ]
