from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphonse.agent.services import JobRunner, JobStore
from alphonse.agent.tools.clock import ClockTool
from alphonse.agent.tools.context_tools import GetMySettingsTool
from alphonse.agent.tools.context_tools import GetUserDetailsTool
from alphonse.agent.tools.domotics_tools import (
    DomoticsExecuteTool,
    DomoticsQueryTool,
    DomoticsSubscribeTool,
)
from alphonse.agent.tools.job_tools import JobCreateTool
from alphonse.agent.tools.job_tools import JobDeleteTool
from alphonse.agent.tools.job_tools import JobListTool
from alphonse.agent.tools.job_tools import JobPauseTool
from alphonse.agent.tools.job_tools import JobResumeTool
from alphonse.agent.tools.job_tools import JobRunNowTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool
from alphonse.agent.tools.memory_tools import GetMissionTool
from alphonse.agent.tools.memory_tools import GetWorkspacePointerTool
from alphonse.agent.tools.memory_tools import ListActiveMissionsTool
from alphonse.agent.tools.memory_tools import SearchEpisodesTool
from alphonse.agent.tools.mcp_call_tool import McpCallTool
from alphonse.agent.tools.scheduler_tool import SchedulerTool
from alphonse.agent.tools.send_message_tool import SendMessageTool
from alphonse.agent.tools.send_message_tool import SendVoiceNoteTool
from alphonse.agent.tools.ssh_terminal_tool import SshTerminalTool
from alphonse.agent.tools.stt_transcribe import SttTranscribeTool
from alphonse.agent.tools.terminal_execute_tool import TerminalExecuteTool
from alphonse.agent.tools.telegram_files import TelegramDownloadFileTool
from alphonse.agent.tools.telegram_files import TelegramGetFileMetaTool
from alphonse.agent.tools.telegram_files import VisionAnalyzeImageTool
from alphonse.agent.tools.telegram_files import VisionExtractTool
from alphonse.agent.tools.user_contact_tools import UserRegisterFromContactTool
from alphonse.agent.tools.user_contact_tools import UserRemoveFromContactTool
from alphonse.agent.tools.user_contact_tools import UserSearchTool
from alphonse.agent.tools.base import ToolDefinition, ToolProtocol
from alphonse.agent.tools.spec import SafetyLevel, ToolSpec


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, definition: ToolDefinition) -> None:
        spec = definition.spec
        executor = definition.executor
        canonical_name = str(spec.canonical_name or "").strip()
        if not canonical_name:
            raise ValueError("tool_definition_missing_canonical_name")
        if not callable(getattr(executor, "execute", None)):
            raise ValueError(f"tool_missing_execute:{canonical_name}")
        keys = [canonical_name]
        keys.extend(str(alias or "").strip() for alias in (spec.aliases or []))
        for key in keys:
            if not key:
                raise ValueError(f"tool_definition_empty_key:{canonical_name}")
            if key in self._tools and self._tools[key] is not definition:
                raise ValueError(f"tool_key_collision:{key}")
            self._tools[key] = definition

    def get(self, key: str) -> ToolDefinition | None:
        return self._tools.get(str(key))

    def keys(self) -> list[str]:
        return sorted(self._tools.keys())

    def definitions(self) -> list[ToolDefinition]:
        seen: set[str] = set()
        out: list[ToolDefinition] = []
        for key in self.keys():
            definition = self._tools.get(key)
            if definition is None:
                continue
            canonical_name = str(definition.spec.canonical_name or "").strip()
            if not canonical_name or canonical_name in seen:
                continue
            seen.add(canonical_name)
            out.append(definition)
        out.sort(key=lambda item: str(item.spec.canonical_name or ""))
        return out


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    spec_by_name = _spec_index()
    job_store = JobStore()
    job_runner = JobRunner(
        job_store=job_store,
        tick_seconds=45,
    )
    runtime_tools = _build_runtime_executors(job_store=job_store, job_runner=job_runner)
    seen_canonical: set[str] = set()
    for executor in runtime_tools:
        canonical_name, _ = _executor_identity(executor)
        if canonical_name in seen_canonical:
            raise ValueError(f"tool_executor_duplicate_canonical_name:{canonical_name}")
        seen_canonical.add(canonical_name)
        spec = _require_spec(spec_by_name, canonical_name)
        if str(spec.canonical_name or "").strip() != canonical_name:
            raise ValueError(
                f"tool_executor_spec_mismatch:{canonical_name}:{str(spec.canonical_name or '').strip()}"
            )
        registry.register(ToolDefinition(spec=spec, executor=executor))
    job_runner.set_tool_registry(registry)
    return registry


def _spec_index() -> dict[str, ToolSpec]:
    out: dict[str, ToolSpec] = {}
    for spec in _default_specs():
        key = str(spec.canonical_name or "").strip()
        if not key:
            continue
        out[key] = spec
    return out


def _require_spec(spec_by_name: dict[str, ToolSpec], canonical_name: str) -> ToolSpec:
    key = str(canonical_name or "").strip()
    spec = spec_by_name.get(key)
    if spec is None:
        raise ValueError(f"tool_spec_missing:{key}")
    return spec


def _build_runtime_executors(*, job_store: JobStore, job_runner: JobRunner) -> list[ToolProtocol]:
    context_clock = ClockTool()
    context_get_my_settings = GetMySettingsTool()
    context_get_user_details = GetUserDetailsTool()
    memory_search_episodes = SearchEpisodesTool()
    memory_get_mission = GetMissionTool()
    memory_list_active_missions = ListActiveMissionsTool()
    memory_get_workspace = GetWorkspacePointerTool()
    communication_send_message = SendMessageTool()
    communication_send_voice_note = SendVoiceNoteTool(_send_message_tool=communication_send_message)
    communication_get_attachment_meta = TelegramGetFileMetaTool()
    communication_get_attachment = TelegramDownloadFileTool()
    audio_speak_local = LocalAudioOutputSpeakTool()
    audio_render_local = LocalAudioOutputRenderTool()
    audio_transcribe = SttTranscribeTool()
    vision_analyze_image = VisionAnalyzeImageTool()
    vision_extract_text = VisionExtractTool()
    terminal_sync = TerminalExecuteTool()
    mcp_call = McpCallTool()
    ssh_terminal = SshTerminalTool()
    scheduler = SchedulerTool()
    job_create = JobCreateTool(job_store)
    job_list = JobListTool(job_store)
    job_pause = JobPauseTool(job_store)
    job_resume = JobResumeTool(job_store)
    job_delete = JobDeleteTool(job_store)
    job_run_now = JobRunNowTool(job_runner)
    user_register_from_contact = UserRegisterFromContactTool()
    user_remove_from_contact = UserRemoveFromContactTool()
    user_search = UserSearchTool()
    domotics_query = DomoticsQueryTool()
    domotics_execute = DomoticsExecuteTool()
    domotics_subscribe = DomoticsSubscribeTool()
    return [
        context_clock,
        context_get_my_settings,
        context_get_user_details,
        memory_search_episodes,
        memory_get_mission,
        memory_list_active_missions,
        memory_get_workspace,
        communication_send_message,
        communication_send_voice_note,
        communication_get_attachment_meta,
        communication_get_attachment,
        audio_speak_local,
        audio_render_local,
        audio_transcribe,
        vision_analyze_image,
        vision_extract_text,
        terminal_sync,
        mcp_call,
        ssh_terminal,
        scheduler,
        job_create,
        job_list,
        job_pause,
        job_resume,
        job_delete,
        job_run_now,
        user_register_from_contact,
        user_remove_from_contact,
        user_search,
        domotics_query,
        domotics_execute,
        domotics_subscribe,
    ]


def _executor_identity(executor: ToolProtocol) -> tuple[str, str]:
    canonical = str(getattr(executor, "canonical_name", "") or "").strip()
    capability = str(getattr(executor, "capability", "") or "").strip()
    if not canonical:
        raise ValueError(f"tool_executor_missing_canonical_name:{type(executor).__name__}")
    if not capability:
        raise ValueError(f"tool_executor_missing_capability:{canonical}")
    return canonical, capability


def planner_visible_tool_definitions(tool_registry: Any) -> list[ToolDefinition]:
    if isinstance(tool_registry, ToolRegistry):
        definitions = tool_registry.definitions()
    else:
        definitions = []
        if hasattr(tool_registry, "keys") and hasattr(tool_registry, "get"):
            seen: set[str] = set()
            for key in list(tool_registry.keys()):  # type: ignore[attr-defined]
                definition = tool_registry.get(key)  # type: ignore[attr-defined]
                if not isinstance(definition, ToolDefinition):
                    continue
                canonical_name = str(definition.spec.canonical_name or "").strip()
                if not canonical_name or canonical_name in seen:
                    continue
                seen.add(canonical_name)
                definitions.append(definition)
        definitions.sort(key=lambda item: str(item.spec.canonical_name or ""))
    return [item for item in definitions if bool(item.spec.visible_to_agent)]


def planner_tool_schemas(tool_registry: Any) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": str(defn.spec.canonical_name or "").strip(),
                "description": str(defn.spec.summary or "").strip() or "Tool available.",
                "parameters": dict(defn.spec.input_schema or {}),
            },
        }
        for defn in planner_visible_tool_definitions(tool_registry)
        if str(defn.spec.canonical_name or "").strip()
    ]


def planner_canonical_tool_names(tool_registry: Any) -> list[str]:
    return sorted(
        {
            str(defn.spec.canonical_name or "").strip()
            for defn in planner_visible_tool_definitions(tool_registry)
            if str(defn.spec.canonical_name or "").strip()
        }
    )


def planner_tool_descriptions(tool_registry: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for defn in planner_visible_tool_definitions(tool_registry):
        name = str(defn.spec.canonical_name or "").strip()
        if not name:
            continue
        out[name] = str(defn.spec.summary or "").strip() or "Tool available."
    return out


def planner_tool_parameters(tool_registry: Any, tool_name: str) -> dict[str, Any] | None:
    requested = str(tool_name or "").strip()
    if not requested:
        return None
    for defn in planner_visible_tool_definitions(tool_registry):
        name = str(defn.spec.canonical_name or "").strip()
        if name == requested:
            return dict(defn.spec.input_schema or {})
    return None
def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _permissive_output_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": True}


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
            canonical_name="communication.send_message",
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
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"To": "Gabriela", "Message": "Hola Gaby, Alex llegará para cenar.", "Channel": "telegram"}],
        ),
        ToolSpec(
            canonical_name="communication.send_voice_note",
            summary="Send a Telegram-style voice note to a recipient.",
            description="Send a Telegram-style voice note to a recipient.",
            when_to_use="Prefer this over communication.send_message when you must deliver a true voice note bubble in Telegram.",
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
            canonical_name="audio.speak_local",
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
            safety_level=SafetyLevel.LOW,
            examples=[{"text": "Hola, te escucho.", "voice": "Ryan"}],
        ),
        ToolSpec(
            canonical_name="audio.render_local",
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
            safety_level=SafetyLevel.LOW,
            examples=[{"text": "Hola Alex", "format": "ogg"}],
        ),
        ToolSpec(
            canonical_name="audio.transcribe",
            summary="Transcribe an audio asset by asset_id into text.",
            description="Transcribe an audio asset by asset_id into text.",
            when_to_use="Use when the incoming message includes an audio asset (including downloaded attachments) and you need its transcript.",
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
            examples=[
                {"asset_id": "asset_123", "language_hint": "es-MX"},
                {"asset_id": "telegram_attachment_audio_001", "language_hint": "en"},
            ],
        ),
        ToolSpec(
            canonical_name="communication.get_attachment_meta",
            summary="Resolve attachment metadata from a provider file_id.",
            description="Resolve attachment metadata from a provider file_id.",
            when_to_use="Use when attachment metadata is required before download/transcription.",
            returns="telegram_file_meta",
            input_schema=_object_schema(
                properties={"file_id": {"type": "string"}},
                required=["file_id"],
            ),
            output_schema=_permissive_output_schema(),
            domain_tags=["communication", "attachments", "files"],
            safety_level=SafetyLevel.MEDIUM,
            examples=[{"file_id": "AgACAgQAAx..."}],
        ),
        ToolSpec(
            canonical_name="communication.get_attachment",
            summary="Download an attachment by provider file_id and return local path details.",
            description="Download an attachment by provider file_id and return local path details.",
            when_to_use="Use when an attachment must be downloaded for downstream processing.",
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
            domain_tags=["communication", "attachments", "files"],
            safety_level=SafetyLevel.HIGH,
            examples=[{"file_id": "AgACAgQAAx...", "sandbox_alias": "telegram"}],
        ),
        ToolSpec(
            canonical_name="vision.analyze_image",
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
            canonical_name="vision.extract_text",
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
            canonical_name="memory.search_episodes",
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
            canonical_name="memory.get_mission",
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
            canonical_name="memory.list_active_missions",
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
            canonical_name="memory.get_workspace",
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
