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
                required=["file_id"],
            ),
            domain_tags=["telegram", "image", "analysis"],
            safety_level=SafetyLevel.HIGH,
            expose_in_catalog=False,
            examples=[{"file_id": "AgACAgQAAx...", "prompt": "Describe the image briefly."}],
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

