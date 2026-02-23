from __future__ import annotations

from typing import Any

from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("io.voice_channel")


class VoiceExtremityAdapter:
    channel_type: str = "voice"

    def __init__(self, *, speaker: Any | None = None) -> None:
        self._speaker = speaker

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        text = str(message.message or "").strip()
        if not text:
            return
        speaker = self._speaker
        if speaker is None:
            # Lazy import prevents import-time cycles with tools registry bootstrap.
            from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool

            speaker = LocalAudioOutputSpeakTool()
            self._speaker = speaker
        result = speaker.execute(text=text, blocking=False)
        status = str((result or {}).get("status") or "").strip().lower()
        if status != "ok":
            error = (result or {}).get("error") if isinstance(result, dict) else None
            code = str((error or {}).get("code") or "voice_delivery_failed")
            logger.warning(
                "VoiceExtremityAdapter failed target=%s correlation_id=%s code=%s",
                message.channel_target,
                message.correlation_id,
                code,
            )
            return
        logger.info(
            "VoiceExtremityAdapter delivered target=%s correlation_id=%s",
            message.channel_target,
            message.correlation_id,
        )
