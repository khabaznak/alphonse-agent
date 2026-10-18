"""Local speech adapters."""

from __future__ import annotations

import base64
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.media_tools_settings import SttSettings, TtsSettings

TTS_RENDER_TOOL_ID = "native.tts_render"
STT_TRANSCRIBE_TOOL_ID = "native.stt_transcribe"
MAX_STT_VERIFICATION_BYTES = 4 * 1024 * 1024
MAX_STT_VERIFICATION_DURATION_MS = 30_000
_STT_VERIFICATION_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
}


def build_tts_render_tool_definition(settings: TtsSettings) -> ToolDefinition:
    descriptor = ToolDescriptor(TTS_RENDER_TOOL_ID, "tts_render", ToolKind.NATIVE, "Render text through the configured local TTS backend.", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, ("media", "tts"), ("native", "media"))
    return ToolDefinition(descriptor, lambda args: render_tts(settings, text=str(args.get("text") or "")), dict(descriptor.argument_schema), enabled=False)


def build_stt_transcribe_tool_definition(settings: SttSettings) -> ToolDefinition:
    descriptor = ToolDescriptor(STT_TRANSCRIBE_TOOL_ID, "stt_transcribe", ToolKind.NATIVE, "Transcribe a supplied v2 audio asset reference.", {"type": "object", "properties": {"asset_path": {"type": "string"}}, "required": ["asset_path"]}, ("media", "stt"), ("native", "media"))
    return ToolDefinition(descriptor, lambda args: transcribe_stt(settings, asset_path=str(args.get("asset_path") or "")), dict(descriptor.argument_schema), enabled=False)


def verify_tts(settings: TtsSettings, *, sample_text: str) -> dict[str, Any]:
    return render_tts(settings, text=sample_text, output_dir=tempfile.mkdtemp(prefix="alphonse-v2-tts-verify-"), allow_say_fallback=False)


def render_tts(settings: TtsSettings, *, text: str, output_dir: str | None = None, allow_say_fallback: bool = True) -> dict[str, Any]:
    rendered = str(text or "").strip()
    if not rendered: return _failed("tts_text_required", "text is required.")
    try:
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        kwargs: dict[str, Any] = {"device_map": settings.device_map or "auto", "local_files_only": settings.local_files_only}
        if settings.attn_implementation: kwargs["attn_implementation"] = settings.attn_implementation
        if settings.dtype:
            import torch
            kwargs["dtype"] = getattr(torch, settings.dtype)
        model = Qwen3TTSModel.from_pretrained(settings.model_id, **kwargs)
        wavs, sample_rate = model.generate_custom_voice(text=rendered, language=settings.language or "Auto", speaker=settings.speaker or "Ryan", instruct=settings.instruct or None)
        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if wav is None: raise RuntimeError("qwen_no_audio")
        root = Path(output_dir or tempfile.mkdtemp(prefix="alphonse-v2-tts-")); root.mkdir(parents=True, exist_ok=True)
        target = root / "speech.wav"; sf.write(str(target), wav, int(sample_rate))
        return _ok({"file_path": str(target), "format": "wav", "mime_type": "audio/wav", "backend": "qwen"})
    except Exception as exc:
        if allow_say_fallback and platform.system() == "Darwin" and shutil.which("say"):
            return _render_say(rendered, output_dir=output_dir, fallback_error=str(exc))
        return _failed("qwen_backend_unavailable", "Qwen TTS could not render audio.", details={"error": str(exc)})


def verify_stt(settings: SttSettings, *, sample_path: str) -> dict[str, Any]:
    return transcribe_stt(settings, asset_path=sample_path)


def verify_stt_recording(
    settings: SttSettings,
    *,
    audio_base64: str,
    mime_type: str,
    duration_ms: int,
) -> dict[str, Any]:
    normalized_mime = str(mime_type or "").strip().lower().split(";", 1)[0]
    suffix = _STT_VERIFICATION_SUFFIXES.get(normalized_mime)
    if suffix is None:
        return _failed("stt_recording_type_unsupported", "The recording format is not supported.")
    if not 0 < int(duration_ms or 0) <= MAX_STT_VERIFICATION_DURATION_MS:
        return _failed("stt_recording_duration_invalid", "Record a message between 1 and 30 seconds.")
    try:
        audio = base64.b64decode(str(audio_base64 or ""), validate=True)
    except (ValueError, TypeError):
        return _failed("stt_recording_invalid", "The recording could not be read.")
    if not audio:
        return _failed("stt_recording_empty", "Record a short message before verifying.")
    if len(audio) > MAX_STT_VERIFICATION_BYTES:
        return _failed("stt_recording_too_large", "The recording is too large; keep it under 30 seconds.")
    with tempfile.TemporaryDirectory(prefix="alphonse-v2-stt-recording-") as root:
        source = Path(root) / f"recording{suffix}"
        source.write_bytes(audio)
        return transcribe_stt(settings, asset_path=str(source))


def transcribe_stt(settings: SttSettings, *, asset_path: str) -> dict[str, Any]:
    source = Path(str(asset_path or "")).expanduser()
    if not source.is_file(): return _failed("stt_sample_not_found", "Audio sample path was not found.")
    command = settings.executable_path or shutil.which("whisper") or ""
    if not command or not Path(command).exists(): return _failed("whisper_cli_not_found", "Whisper CLI was not found.")
    with tempfile.TemporaryDirectory(prefix="alphonse-v2-stt-") as target:
        cmd = [command, str(source), "--model", settings.model, "--output_dir", target, "--output_format", "json"]
        if settings.default_language: cmd.extend(["--language", settings.default_language.split("-", 1)[0]])
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode != 0: return _failed("whisper_transcription_failed", "Whisper could not transcribe the sample.", details={"stderr": str(completed.stderr or "")[-1000:]})
        outputs = sorted(Path(target).glob("*.json"))
        if not outputs: return _failed("whisper_output_missing", "Whisper did not create a transcript.")
        try: body = json.loads(outputs[0].read_text(encoding="utf-8"))
        except Exception: return _failed("whisper_output_invalid", "Whisper returned an invalid transcript.")
        text = str(body.get("text") or "").strip()
        if not text: return _failed("transcript_empty", "Whisper returned an empty transcript.")
        return _ok({"text": text, "segments": body.get("segments") if isinstance(body.get("segments"), list) else [], "model": settings.model})


def _render_say(text: str, *, output_dir: str | None, fallback_error: str) -> dict[str, Any]:
    root = Path(output_dir or tempfile.mkdtemp(prefix="alphonse-v2-say-")); root.mkdir(parents=True, exist_ok=True)
    target = root / "speech.aiff"
    completed = subprocess.run(["say", "-o", str(target), text], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0: return _failed("say_fallback_failed", "macOS say fallback could not render audio.")
    return _ok({"file_path": str(target), "format": "aiff", "mime_type": "audio/aiff", "backend": "say", "fallback_from": "qwen", "fallback_error": fallback_error})
def _ok(output: dict[str, Any]) -> dict[str, Any]: return {"output": output, "exception": None}
def _failed(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]: return {"output": None, "exception": {"code": code, "message": message, "details": details or {}}}
