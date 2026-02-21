from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphonse.agent.services import JobRunner, JobStore, ScratchpadService
from alphonse.agent.tools.clock import ClockTool
from alphonse.agent.tools.job_tools import JobCreateTool
from alphonse.agent.tools.job_tools import JobDeleteTool
from alphonse.agent.tools.job_tools import JobListTool
from alphonse.agent.tools.job_tools import JobPauseTool
from alphonse.agent.tools.job_tools import JobResumeTool
from alphonse.agent.tools.job_tools import JobRunNowTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadAppendTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadCreateTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadForkTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadListTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadReadTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadSearchTool
from alphonse.agent.tools.scheduler_tool import SchedulerTool
from alphonse.agent.tools.stt_transcribe import SttTranscribeTool
from alphonse.agent.tools.terminal_execute_tool import TerminalExecuteTool
from alphonse.agent.tools.telegram_files import AnalyzeTelegramImageTool
from alphonse.agent.tools.telegram_files import TelegramDownloadFileTool
from alphonse.agent.tools.telegram_files import TelegramGetFileMetaTool
from alphonse.agent.tools.telegram_files import TranscribeTelegramAudioTool
from alphonse.agent.tools.telegram_files import VisionAnalyzeImageTool
from alphonse.agent.tools.user_contact_tools import UserRegisterFromContactTool
from alphonse.agent.tools.user_contact_tools import UserRemoveFromContactTool

from alphonse.agent.tools.subprocess import SubprocessTool


@dataclass
class ToolRegistry:
    _tools: dict[str, Any] = field(default_factory=dict)

    def register(self, key: str, tool: Any) -> None:
        self._tools[str(key)] = tool

    def get(self, key: str) -> Any | None:
        return self._tools.get(str(key))

    def keys(self) -> list[str]:
        return sorted(self._tools.keys())


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    clock = ClockTool()
    scheduler = SchedulerTool()
    local_audio_output = LocalAudioOutputSpeakTool()
    stt_transcribe = SttTranscribeTool()
    telegram_get_file = TelegramGetFileMetaTool()
    telegram_download_file = TelegramDownloadFileTool()
    transcribe_audio = TranscribeTelegramAudioTool()
    analyze_image = AnalyzeTelegramImageTool()
    vision_analyze_image = VisionAnalyzeImageTool()
    python_subprocess = SubprocessTool()
    terminal_execute = TerminalExecuteTool()
    scratchpad_service = ScratchpadService()
    scratchpad_create = ScratchpadCreateTool(scratchpad_service)
    scratchpad_append = ScratchpadAppendTool(scratchpad_service)
    scratchpad_read = ScratchpadReadTool(scratchpad_service)
    scratchpad_list = ScratchpadListTool(scratchpad_service)
    scratchpad_search = ScratchpadSearchTool(scratchpad_service)
    scratchpad_fork = ScratchpadForkTool(scratchpad_service)
    job_store = JobStore()
    job_runner = JobRunner(
        job_store=job_store,
        scratchpad_service=scratchpad_service,
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
    registry.register("getTime", clock)
    registry.register("createReminder", scheduler)
    registry.register("local_audio_output.speak", local_audio_output)
    registry.register("stt_transcribe", stt_transcribe)
    registry.register("telegramGetFileMeta", telegram_get_file)
    registry.register("telegramDownloadFile", telegram_download_file)
    registry.register("transcribeTelegramAudio", transcribe_audio)
    registry.register("analyzeTelegramImage", analyze_image)
    registry.register("vision_analyze_image", vision_analyze_image)
    # Internal aliases kept for compatibility.
    registry.register("clock", clock)
    registry.register("python_subprocess", python_subprocess)
    registry.register("terminal_execute", terminal_execute)
    registry.register("scratchpad_create", scratchpad_create)
    registry.register("scratchpad_append", scratchpad_append)
    registry.register("scratchpad_read", scratchpad_read)
    registry.register("scratchpad_list", scratchpad_list)
    registry.register("scratchpad_search", scratchpad_search)
    registry.register("scratchpad_fork", scratchpad_fork)
    registry.register("job_create", job_create)
    registry.register("job_list", job_list)
    registry.register("job_pause", job_pause)
    registry.register("job_resume", job_resume)
    registry.register("job_delete", job_delete)
    registry.register("job_run_now", job_run_now)
    registry.register("user_register_from_contact", user_register_from_contact)
    registry.register("user_remove_from_contact", user_remove_from_contact)
    job_runner.set_tool_registry(registry)
    return registry
