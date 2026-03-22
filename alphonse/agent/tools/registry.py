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
from alphonse.agent.tools.telegram_files import TranscribeTelegramAudioTool
from alphonse.agent.tools.telegram_files import VisionAnalyzeImageTool
from alphonse.agent.tools.telegram_files import VisionExtractTool
from alphonse.agent.tools.user_contact_tools import UserRegisterFromContactTool
from alphonse.agent.tools.user_contact_tools import UserRemoveFromContactTool
from alphonse.agent.tools.user_contact_tools import UserSearchTool
from alphonse.agent.tools.base import ToolDefinition, ToolProtocol
from alphonse.agent.tools.registry2 import build_planner_tool_registry
from alphonse.agent.tools.spec import ToolSpec


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


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    spec_by_name = _spec_index()
    job_store = JobStore()
    job_runner = JobRunner(
        job_store=job_store,
        tick_seconds=45,
    )
    runtime_tools = _build_runtime_executors(job_store=job_store, job_runner=job_runner)
    for canonical_name, executor in runtime_tools:
        spec = _require_spec(spec_by_name, canonical_name)
        registry.register(ToolDefinition(spec=spec, executor=executor))
    job_runner.set_tool_registry(registry)
    return registry


def _spec_index() -> dict[str, ToolSpec]:
    out: dict[str, ToolSpec] = {}
    for spec in build_planner_tool_registry().specs():
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


def _build_runtime_executors(*, job_store: JobStore, job_runner: JobRunner) -> list[tuple[str, ToolProtocol]]:
    clock = ClockTool()
    get_my_settings = GetMySettingsTool()
    get_user_details = GetUserDetailsTool()
    search_episodes = SearchEpisodesTool()
    get_mission = GetMissionTool()
    list_active_missions = ListActiveMissionsTool()
    get_workspace_pointer = GetWorkspacePointerTool()
    scheduler = SchedulerTool()
    send_message = SendMessageTool()
    send_voice_note = SendVoiceNoteTool(_send_message_tool=send_message)
    local_audio_output = LocalAudioOutputSpeakTool()
    local_audio_render = LocalAudioOutputRenderTool()
    stt_transcribe = SttTranscribeTool()
    telegram_get_file = TelegramGetFileMetaTool()
    telegram_download_file = TelegramDownloadFileTool()
    transcribe_audio = TranscribeTelegramAudioTool()
    vision_analyze_image = VisionAnalyzeImageTool()
    vision_extract = VisionExtractTool()
    terminal_sync = TerminalExecuteTool()
    mcp_call = McpCallTool()
    ssh_terminal = SshTerminalTool()
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
        ("get_time", clock),
        ("create_reminder", scheduler),
        ("get_my_settings", get_my_settings),
        ("get_user_details", get_user_details),
        ("search_episodes", search_episodes),
        ("get_mission", get_mission),
        ("list_active_missions", list_active_missions),
        ("get_workspace_pointer", get_workspace_pointer),
        ("send_message", send_message),
        ("send_voice_note", send_voice_note),
        ("local_audio_output_speak", local_audio_output),
        ("local_audio_output_render", local_audio_render),
        ("stt_transcribe", stt_transcribe),
        ("telegram_get_file_meta", telegram_get_file),
        ("telegram_download_file", telegram_download_file),
        ("transcribe_telegram_audio", transcribe_audio),
        ("vision_analyze_image", vision_analyze_image),
        ("vision_extract", vision_extract),
        ("terminal_sync", terminal_sync),
        ("mcp_call", mcp_call),
        ("ssh_terminal", ssh_terminal),
        ("job_create", job_create),
        ("job_list", job_list),
        ("job_pause", job_pause),
        ("job_resume", job_resume),
        ("job_delete", job_delete),
        ("job_run_now", job_run_now),
        ("user_register_from_contact", user_register_from_contact),
        ("user_remove_from_contact", user_remove_from_contact),
        ("user_search", user_search),
        ("domotics.query", domotics_query),
        ("domotics.execute", domotics_execute),
        ("domotics.subscribe", domotics_subscribe),
    ]
