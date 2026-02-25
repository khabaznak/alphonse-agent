from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphonse.agent.services import JobRunner, JobStore
from alphonse.agent.tools.clock import ClockTool
from alphonse.agent.tools.job_tools import JobCreateTool
from alphonse.agent.tools.job_tools import JobDeleteTool
from alphonse.agent.tools.job_tools import JobListTool
from alphonse.agent.tools.job_tools import JobPauseTool
from alphonse.agent.tools.job_tools import JobResumeTool
from alphonse.agent.tools.job_tools import JobRunNowTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool
from alphonse.agent.tools.scheduler_tool import SchedulerTool
from alphonse.agent.tools.send_message_tool import SendMessageTool
from alphonse.agent.tools.send_message_tool import SendVoiceNoteTool
from alphonse.agent.tools.ssh_terminal_tool import SshTerminalTool
from alphonse.agent.tools.stt_transcribe import SttTranscribeTool
from alphonse.agent.tools.terminal_execute_tool import TerminalExecuteTool
from alphonse.agent.tools.telegram_files import AnalyzeTelegramImageTool
from alphonse.agent.tools.telegram_files import TelegramDownloadFileTool
from alphonse.agent.tools.telegram_files import TelegramGetFileMetaTool
from alphonse.agent.tools.telegram_files import TranscribeTelegramAudioTool
from alphonse.agent.tools.telegram_files import VisionAnalyzeImageTool
from alphonse.agent.tools.terminal_async_tools import TerminalCommandStatusTool
from alphonse.agent.tools.terminal_async_tools import TerminalCommandSubmitTool
from alphonse.agent.tools.user_contact_tools import UserRegisterFromContactTool
from alphonse.agent.tools.user_contact_tools import UserRemoveFromContactTool
from alphonse.agent.tools.user_contact_tools import UserSearchTool
from alphonse.agent.tools.base import ToolProtocol


@dataclass
class ToolRegistry:
    _tools: dict[str, Any] = field(default_factory=dict)

    def register(self, key: str, tool: ToolProtocol) -> None:
        if not callable(getattr(tool, "execute", None)):
            raise ValueError(f"tool_missing_execute:{key}")
        self._tools[str(key)] = tool

    def get(self, key: str) -> ToolProtocol | None:
        return self._tools.get(str(key))

    def keys(self) -> list[str]:
        return sorted(self._tools.keys())


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    clock = ClockTool()
    scheduler = SchedulerTool()
    send_message = SendMessageTool()
    send_voice_note = SendVoiceNoteTool(_send_message_tool=send_message)
    local_audio_output = LocalAudioOutputSpeakTool()
    local_audio_render = LocalAudioOutputRenderTool()
    stt_transcribe = SttTranscribeTool()
    telegram_get_file = TelegramGetFileMetaTool()
    telegram_download_file = TelegramDownloadFileTool()
    transcribe_audio = TranscribeTelegramAudioTool()
    analyze_image = AnalyzeTelegramImageTool()
    vision_analyze_image = VisionAnalyzeImageTool()
    terminal_sync = TerminalExecuteTool()
    terminal_async = TerminalCommandSubmitTool()
    terminal_async_command_status = TerminalCommandStatusTool()
    ssh_terminal = SshTerminalTool()
    job_store = JobStore()
    job_runner = JobRunner(
        job_store=job_store,
        tick_seconds=45,
    )
    job_create = JobCreateTool(job_store)
    job_list = JobListTool(job_store)
    job_pause = JobPauseTool(job_store)
    job_resume = JobResumeTool(job_store)
    job_delete = JobDeleteTool(job_store)
    job_run_now = JobRunNowTool(job_runner)
    user_register_from_contact = UserRegisterFromContactTool()
    user_remove_from_contact = UserRemoveFromContactTool()
    user_search = UserSearchTool()
    registry.register("get_time", clock)
    registry.register("create_reminder", scheduler)
    registry.register("send_message", send_message)
    registry.register("send_voice_note", send_voice_note)
    registry.register("local_audio_output_speak", local_audio_output)
    registry.register("local_audio_output_render", local_audio_render)
    registry.register("stt_transcribe", stt_transcribe)
    registry.register("telegram_get_file_meta", telegram_get_file)
    registry.register("telegram_download_file", telegram_download_file)
    registry.register("transcribe_telegram_audio", transcribe_audio)
    registry.register("analyze_telegram_image", analyze_image)
    registry.register("vision_analyze_image", vision_analyze_image)
    # Internal aliases kept for compatibility.
    registry.register("clock", clock)
    registry.register("getTime", clock)
    registry.register("createReminder", scheduler)
    registry.register("sendMessage", send_message)
    registry.register("sendVoiceNote", send_voice_note)
    registry.register("local_audio_output.speak", local_audio_output)
    registry.register("local_audio_output.render", local_audio_render)
    registry.register("telegramGetFileMeta", telegram_get_file)
    registry.register("telegramDownloadFile", telegram_download_file)
    registry.register("transcribeTelegramAudio", transcribe_audio)
    registry.register("analyzeTelegramImage", analyze_image)
    registry.register("terminal_sync", terminal_sync)
    registry.register("terminal_async", terminal_async)
    registry.register("terminal_async_command_status", terminal_async_command_status)
    registry.register("ssh_terminal", ssh_terminal)
    registry.register("job_create", job_create)
    registry.register("job_list", job_list)
    registry.register("job_pause", job_pause)
    registry.register("job_resume", job_resume)
    registry.register("job_delete", job_delete)
    registry.register("job_run_now", job_run_now)
    registry.register("user_register_from_contact", user_register_from_contact)
    registry.register("user_remove_from_contact", user_remove_from_contact)
    registry.register("user_search", user_search)
    job_runner.set_tool_registry(registry)
    return registry
