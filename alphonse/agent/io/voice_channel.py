from __future__ import annotations

import logging

from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool

logger = logging.getLogger(__name__)


class VoiceExtremityAdapter:
    channel_type: str = "voice"

    def __init__(self, *, speaker: LocalAudioOutputSpeakTool | None = None) -> None:
        self._speaker = speaker or LocalAudioOutputSpeakTool()

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        text = str(message.message or "").strip()
        if not text:
            return
        result = self._speaker.execute(text=text, blocking=False)
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
