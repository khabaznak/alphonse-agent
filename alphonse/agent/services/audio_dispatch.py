from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.plans import CortexPlan
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool

logger = get_component_logger("services.audio_dispatch")


def maybe_emit_local_audio_reply(*, payload: dict[str, object], reply_text: str, correlation_id: str) -> None:
    if extract_audio_mode(payload) != "local_audio":
        return
    spoken = str(reply_text or "").strip()
    if not spoken:
        return
    try:
        result = LocalAudioOutputSpeakTool().execute(text=spoken, blocking=False)
        logger.info(
            "local_audio_output correlation_id=%s status=%s",
            correlation_id,
            str(result.get("status") or "unknown"),
        )
    except Exception:
        logger.exception(
            "local_audio_output_failed correlation_id=%s",
            correlation_id,
        )


def extract_tts_transcript(plans: list[Any]) -> str:
    for item in plans:
        if not isinstance(item, CortexPlan):
            continue
        tool_name = str(item.tool or "").strip().lower()
        params = dict(item.parameters or {})
        payload = dict(item.payload or {})
        merged = dict(payload)
        merged.update(params)
        transcript = ""
        if tool_name in {"local_audio_output.speak", "local_audio_output_speak"}:
            transcript = str(merged.get("text") or "").strip()
        elif tool_name == "sendvoicenote":
            transcript = str(merged.get("message") or merged.get("caption") or "").strip()
        elif tool_name == "sendmessage":
            delivery_mode = str(merged.get("delivery_mode") or "").strip().lower()
            as_voice = bool(merged.get("as_voice") is True)
            if delivery_mode == "audio" or as_voice:
                transcript = str(merged.get("message") or merged.get("caption") or "").strip()
        if transcript:
            return transcript
    return ""


def extract_audio_mode(payload: dict[str, object]) -> str:
    controls = payload.get("controls")
    if isinstance(controls, dict):
        mode = str(controls.get("audio_mode") or "").strip().lower()
        if mode:
            return mode
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        inner_controls = metadata.get("controls")
        if isinstance(inner_controls, dict):
            mode = str(inner_controls.get("audio_mode") or "").strip().lower()
            if mode:
                return mode
        raw = metadata.get("raw")
        if isinstance(raw, dict):
            raw_controls = raw.get("controls")
            if isinstance(raw_controls, dict):
                mode = str(raw_controls.get("audio_mode") or "").strip().lower()
                if mode:
                    return mode
    provider_event = payload.get("provider_event")
    if isinstance(provider_event, dict):
        event_controls = provider_event.get("controls")
        if isinstance(event_controls, dict):
            mode = str(event_controls.get("audio_mode") or "").strip().lower()
            if mode:
                return mode
    return "none"
