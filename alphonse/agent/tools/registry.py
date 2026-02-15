from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphonse.agent.tools.clock import ClockTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool
from alphonse.agent.tools.scheduler import SchedulerTool
from alphonse.agent.tools.stt_transcribe import SttTranscribeTool
from alphonse.agent.tools.telegram_files import AnalyzeTelegramImageTool
from alphonse.agent.tools.telegram_files import TelegramDownloadFileTool
from alphonse.agent.tools.telegram_files import TelegramGetFileMetaTool
from alphonse.agent.tools.telegram_files import TranscribeTelegramAudioTool


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
    return registry
