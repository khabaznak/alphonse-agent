from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphonse.agent.services import ScratchpadService
from alphonse.agent.tools.clock import ClockTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadAppendTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadCreateTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadForkTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadListTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadReadTool
from alphonse.agent.tools.scratchpad_tools import ScratchpadSearchTool
from alphonse.agent.tools.scheduler import SchedulerTool
from alphonse.agent.tools.stt_transcribe import SttTranscribeTool
from alphonse.agent.tools.telegram_files import AnalyzeTelegramImageTool
from alphonse.agent.tools.telegram_files import TelegramDownloadFileTool
from alphonse.agent.tools.telegram_files import TelegramGetFileMetaTool
from alphonse.agent.tools.telegram_files import TranscribeTelegramAudioTool

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
    python_subprocess = SubprocessTool()
    scratchpad_service = ScratchpadService()
    scratchpad_create = ScratchpadCreateTool(scratchpad_service)
    scratchpad_append = ScratchpadAppendTool(scratchpad_service)
    scratchpad_read = ScratchpadReadTool(scratchpad_service)
    scratchpad_list = ScratchpadListTool(scratchpad_service)
    scratchpad_search = ScratchpadSearchTool(scratchpad_service)
    scratchpad_fork = ScratchpadForkTool(scratchpad_service)
    registry.register("getTime", clock)
    registry.register("createReminder", scheduler)
    registry.register("createTimeEventTrigger", scheduler)
    registry.register("local_audio_output.speak", local_audio_output)
    registry.register("stt_transcribe", stt_transcribe)
    registry.register("telegramGetFileMeta", telegram_get_file)
    registry.register("telegramDownloadFile", telegram_download_file)
    registry.register("transcribeTelegramAudio", transcribe_audio)
    registry.register("analyzeTelegramImage", analyze_image)
    # Internal aliases kept for compatibility.
    registry.register("clock", clock)
    registry.register("schedule_event", scheduler)
    registry.register("python_subprocess", python_subprocess)
    registry.register("scratchpad_create", scratchpad_create)
    registry.register("scratchpad_append", scratchpad_append)
    registry.register("scratchpad_read", scratchpad_read)
    registry.register("scratchpad_list", scratchpad_list)
    registry.register("scratchpad_search", scratchpad_search)
    registry.register("scratchpad_fork", scratchpad_fork)
    return registry
