from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import shutil
from alphonse.agent.observability.log_manager import get_component_logger
import platform
import subprocess
import threading
from typing import Any
from dataclasses import dataclass

from alphonse.agent.nervous_system.sandbox_dirs import (
    PRIMARY_WORKDIR_ALIASES,
    default_sandbox_root,
    get_sandbox_alias,
)
from alphonse.agent.nervous_system.voice_profiles import (
    get_default_voice_profile,
    resolve_voice_profile,
)

logger = get_component_logger("tools.local_audio_output")


@dataclass
class _VoiceSelection:
    requested_voice: str
    speaker: str
    instruct: str | None = None
    sample_path: str | None = None
    profile_id: str | None = None
    profile_name: str | None = None
    is_profile: bool = False


class _TTSBackend:
    def speak(
        self,
        *,
        text: str,
        voice: str,
        blocking: bool,
        volume: float | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def render(
        self,
        *,
        text: str,
        voice: str,
        output_dir: str | None,
        filename_prefix: str,
        format: str,
    ) -> dict[str, Any]:
        raise NotImplementedError


class _SayBackend(_TTSBackend):
    def speak(
        self,
        *,
        text: str,
        voice: str,
        blocking: bool,
        volume: float | None,
    ) -> dict[str, Any]:
        _ = volume
        spoken_text = str(text or "").strip()
        selected_voice = str(voice or "default").strip() or "default"
        is_blocking = bool(blocking)
        preview = spoken_text[:40] + ("..." if len(spoken_text) > 40 else "")

        logger.info(
            "local_audio_output_speak invoke backend=say text_len=%s voice=%s blocking=%s preview=%s",
            len(spoken_text),
            selected_voice,
            is_blocking,
            preview,
        )
        if not spoken_text:
            return _failed("text_required", "text is required")

        if platform.system() != "Darwin":
            return _failed(
                "local_audio_output_not_supported",
                "local_audio_output_speak is currently supported only on macOS with the say backend.",
            )

        cmd = _build_say_command(text=spoken_text, voice=selected_voice)
        if is_blocking:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                stderr = str(completed.stderr or "").strip()
                logger.error(
                    "local_audio_output_speak failed backend=say returncode=%s stderr=%s",
                    completed.returncode,
                    stderr,
                )
                return _failed("tts_command_failed", "tts command failed", details={"stderr": stderr})
            logger.info("local_audio_output_speak success backend=say mode=blocking")
            return _ok({"mode": "blocking", "backend": "say"})

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        watcher = threading.Thread(
            target=_watch_say_process,
            args=(proc, preview),
            daemon=True,
        )
        watcher.start()
        logger.info("local_audio_output_speak success backend=say mode=non_blocking pid=%s", proc.pid)
        return _ok({"mode": "non_blocking", "pid": int(proc.pid), "backend": "say"})

    def render(
        self,
        *,
        text: str,
        voice: str,
        output_dir: str | None,
        filename_prefix: str,
        format: str,
    ) -> dict[str, Any]:
        spoken_text = str(text or "").strip()
        if not spoken_text:
            return _failed("text_required", "text is required", tool="local_audio_output_render")
        if platform.system() != "Darwin":
            return _failed(
                "local_audio_output_not_supported",
                "local_audio_output_render is currently supported only on macOS with the say backend.",
                tool="local_audio_output_render",
            )
        selected_voice = str(voice or "default").strip() or "default"
        target_format = str(format or "m4a").strip().lower()
        if target_format not in {"aiff", "m4a", "ogg"}:
            return _failed(
                "unsupported_audio_format",
                "Supported formats are: aiff, m4a, ogg",
                tool="local_audio_output_render",
            )
        root = _resolve_render_output_dir(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        prefix = str(filename_prefix or "response").strip() or "response"
        source_path = root / f"{prefix}-{stamp}.aiff"
        say_cmd = _build_say_command(text=spoken_text, voice=selected_voice, output_path=source_path)
        completed = subprocess.run(
            say_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = str(completed.stderr or "").strip()
            return _failed(
                "tts_command_failed",
                "tts command failed",
                details={"stderr": stderr},
                tool="local_audio_output_render",
            )
        if target_format == "aiff":
            cleanup = _cleanup_rendered_audio(root=root, keep_path=source_path)
            return _ok(
                {
                    "file_path": str(source_path),
                    "format": "aiff",
                    "mime_type": "audio/aiff",
                    "retention": cleanup,
                    "backend": "say",
                },
                tool="local_audio_output_render",
            )
        if target_format == "m4a":
            target_path = source_path.with_suffix(".m4a")
            converted = subprocess.run(
                [
                    "afconvert",
                    "-f",
                    "m4af",
                    "-d",
                    "aac",
                    str(source_path),
                    str(target_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if converted.returncode != 0:
                stderr = str(converted.stderr or "").strip()
                return _failed(
                    "audio_convert_failed",
                    "failed converting aiff to m4a",
                    details={"stderr": stderr},
                    tool="local_audio_output_render",
                )
            mime_type = "audio/mp4"
        else:
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                return _failed(
                    "ffmpeg_not_installed",
                    "ffmpeg is required to convert audio to ogg/opus",
                    tool="local_audio_output_render",
                )
            target_path = source_path.with_suffix(".ogg")
            converted = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    str(source_path),
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "32k",
                    str(target_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if converted.returncode != 0:
                stderr = str(converted.stderr or "").strip()
                return _failed(
                    "audio_convert_failed",
                    "failed converting aiff to ogg",
                    details={"stderr": stderr},
                    tool="local_audio_output_render",
                )
            mime_type = "audio/ogg"
        try:
            source_path.unlink(missing_ok=True)
        except Exception:
            pass
        cleanup = _cleanup_rendered_audio(root=root, keep_path=target_path)
        return _ok(
            {
                "file_path": str(target_path),
                "format": target_format,
                "mime_type": mime_type,
                "retention": cleanup,
                "backend": "say",
            },
            tool="local_audio_output_render",
        )


class _QwenBackend(_TTSBackend):
    def __init__(self) -> None:
        self._model = None
        self._soundfile = None

    def speak(
        self,
        *,
        text: str,
        voice: str,
        blocking: bool,
        volume: float | None,
    ) -> dict[str, Any]:
        _ = volume
        spoken_text = str(text or "").strip()
        if not spoken_text:
            return _failed("text_required", "text is required")
        rendered = self.render(
            text=spoken_text,
            voice=voice,
            output_dir=None,
            filename_prefix="local-speak",
            format="m4a",
        )
        if rendered.get("exception") is not None:
            return rendered
        output = rendered.get("output") if isinstance(rendered.get("output"), dict) else {}
        file_path = str(output.get("file_path") or "").strip()
        if not file_path:
            return _failed("tts_render_missing_path", "tts render did not produce an audio file")
        player = _resolve_audio_player(file_path)
        if player is None:
            return _failed(
                "audio_player_not_available",
                "No supported local audio player found. Install ffplay or use macOS afplay.",
            )
        cmd, player_name = player
        if blocking:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                stderr = str(completed.stderr or "").strip()
                return _failed("audio_player_failed", "local audio playback failed", details={"stderr": stderr})
            return _ok({"mode": "blocking", "backend": "qwen", "player": player_name})
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return _ok({"mode": "non_blocking", "pid": int(proc.pid), "backend": "qwen", "player": player_name})

    def render(
        self,
        *,
        text: str,
        voice: str,
        output_dir: str | None,
        filename_prefix: str,
        format: str,
    ) -> dict[str, Any]:
        spoken_text = str(text or "").strip()
        if not spoken_text:
            return _failed("text_required", "text is required", tool="local_audio_output_render")

        target_format = str(format or "m4a").strip().lower()
        if target_format not in {"aiff", "m4a", "ogg"}:
            return _failed(
                "unsupported_audio_format",
                "Supported formats are: aiff, m4a, ogg",
                tool="local_audio_output_render",
            )

        deps = self._ensure_deps()
        if deps is not None:
            return _failed("qwen_backend_unavailable", deps, tool="local_audio_output_render")

        root = _resolve_render_output_dir(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        prefix = str(filename_prefix or "response").strip() or "response"
        wav_path = root / f"{prefix}-{stamp}.wav"

        model = self._load_model()
        if model is None:
            return _failed(
                "qwen_model_load_failed",
                "Unable to load Qwen TTS model. Verify model id and runtime dependencies.",
                tool="local_audio_output_render",
            )

        language = str(os.getenv("ALPHONSE_QWEN_TTS_LANGUAGE") or "Auto").strip() or "Auto"
        selection = _resolve_voice_selection(voice)

        try:
            wavs, sample_rate = _generate_qwen_custom_voice(
                model=model,
                text=spoken_text,
                language=language,
                speaker=selection.speaker,
                instruct=selection.instruct,
                sample_path=selection.sample_path,
            )
        except Exception as exc:
            return _failed(
                "qwen_generate_failed",
                "Qwen TTS synthesis failed",
                details={"error": str(exc)},
                tool="local_audio_output_render",
            )

        wav = _first_wav(wavs)
        if wav is None:
            return _failed(
                "qwen_generate_failed",
                "Qwen TTS synthesis returned no audio frames",
                tool="local_audio_output_render",
            )
        try:
            self._soundfile.write(str(wav_path), wav, int(sample_rate))
        except Exception as exc:
            return _failed(
                "qwen_write_failed",
                "Failed writing synthesized audio",
                details={"error": str(exc)},
                tool="local_audio_output_render",
            )

        conversion = _convert_from_wav(source_path=wav_path, target_format=target_format)
        if conversion.get("exception") is not None:
            return conversion

        target_path = Path(str((conversion.get("output") or {}).get("file_path") or wav_path))
        cleanup = _cleanup_rendered_audio(root=root, keep_path=target_path)
        payload = dict(conversion.get("output") or {})
        payload["retention"] = cleanup
        payload["backend"] = "qwen"
        if selection.is_profile:
            payload["voice_profile_id"] = selection.profile_id
            payload["voice_profile_name"] = selection.profile_name
        return _ok(payload, tool="local_audio_output_render")

    def _ensure_deps(self) -> str | None:
        if self._soundfile is not None:
            return None
        try:
            import soundfile as sf
            from qwen_tts import Qwen3TTSModel

            self._soundfile = sf
            self._model_cls = Qwen3TTSModel
            return None
        except Exception:
            return (
                "Qwen backend dependencies are unavailable. Install with: pip install -U qwen-tts soundfile"
            )

    def _load_model(self):
        if self._model is not None:
            return self._model
        model_id = str(os.getenv("ALPHONSE_QWEN_TTS_MODEL") or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").strip()
        kwargs: dict[str, Any] = {
            "device_map": str(os.getenv("ALPHONSE_QWEN_TTS_DEVICE_MAP") or "auto").strip() or "auto",
        }
        attn = str(os.getenv("ALPHONSE_QWEN_TTS_ATTN_IMPLEMENTATION") or "").strip()
        if attn:
            kwargs["attn_implementation"] = attn
        dtype = str(os.getenv("ALPHONSE_QWEN_TTS_DTYPE") or "").strip().lower()
        if dtype in {"bfloat16", "float16", "float32"}:
            try:
                import torch

                kwargs["dtype"] = getattr(torch, dtype)
            except Exception:
                pass
        try:
            self._model = self._model_cls.from_pretrained(model_id, **kwargs)
            return self._model
        except Exception as exc:
            logger.error("local_audio_output_qwen model_load_failed model=%s error=%s", model_id, str(exc))
            return None


class LocalAudioOutputSpeakTool:
    """Speak text out loud on the local machine using selected TTS backend."""
    canonical_name: str = "local_audio_output_speak"
    capability: str = "communication"

    def execute(
        self,
        *,
        text: str,
        voice: str = "default",
        blocking: bool = False,
        volume: float | None = None,
    ) -> dict[str, Any]:
        backend = _resolve_tts_backend()
        result = backend.speak(text=text, voice=voice, blocking=blocking, volume=volume)
        if not isinstance(backend, _QwenBackend):
            return result
        if not isinstance(result.get("exception"), dict):
            return result
        fallback_voice = _resolve_say_fallback_voice(requested_voice=voice)
        code = str((result.get("exception") or {}).get("code") or "unknown")
        logger.warning(
            "local_audio_output_qwen_fallback tool=speak code=%s fallback_backend=say requested_voice=%s fallback_voice=%s",
            code,
            str(voice or "default"),
            fallback_voice,
        )
        fallback = _SayBackend().speak(text=text, voice=fallback_voice, blocking=blocking, volume=volume)
        return _with_fallback_metadata(fallback, code=code)


class LocalAudioOutputRenderTool:
    """Render text-to-speech to an audio file for downstream integrations."""
    canonical_name: str = "local_audio_output_render"
    capability: str = "communication"

    def execute(
        self,
        *,
        text: str,
        voice: str = "default",
        output_dir: str | None = None,
        filename_prefix: str = "response",
        format: str = "m4a",
    ) -> dict[str, Any]:
        backend = _resolve_tts_backend()
        result = backend.render(
            text=text,
            voice=voice,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            format=format,
        )
        if not isinstance(backend, _QwenBackend):
            return result
        if not isinstance(result.get("exception"), dict):
            return result
        fallback_voice = _resolve_say_fallback_voice(requested_voice=voice)
        code = str((result.get("exception") or {}).get("code") or "unknown")
        logger.warning(
            "local_audio_output_qwen_fallback tool=render code=%s fallback_backend=say requested_voice=%s fallback_voice=%s",
            code,
            str(voice or "default"),
            fallback_voice,
        )
        fallback = _SayBackend().render(
            text=text,
            voice=fallback_voice,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            format=format,
        )
        return _with_fallback_metadata(fallback, code=code)


def _resolve_tts_backend() -> _TTSBackend:
    backend = str(os.getenv("ALPHONSE_TTS_BACKEND") or "say").strip().lower()
    if backend == "qwen":
        return _QwenBackend()
    return _SayBackend()


def _resolve_voice_selection(voice: str) -> _VoiceSelection:
    requested = str(voice or "default").strip() or "default"
    if requested.lower() != "default":
        profile = resolve_voice_profile(requested)
        if isinstance(profile, dict):
            return _selection_from_profile(profile, requested_voice=requested)
        return _VoiceSelection(
            requested_voice=requested,
            speaker=requested,
            instruct=str(os.getenv("ALPHONSE_QWEN_TTS_INSTRUCT") or "").strip() or None,
        )
    profile = get_default_voice_profile()
    if isinstance(profile, dict):
        return _selection_from_profile(profile, requested_voice="default")
    return _VoiceSelection(
        requested_voice="default",
        speaker=_resolve_qwen_speaker("default"),
        instruct=str(os.getenv("ALPHONSE_QWEN_TTS_INSTRUCT") or "").strip() or None,
    )


def _selection_from_profile(profile: dict[str, Any], *, requested_voice: str) -> _VoiceSelection:
    speaker_hint = str(profile.get("speaker_hint") or "").strip()
    profile_name = str(profile.get("name") or "").strip()
    return _VoiceSelection(
        requested_voice=requested_voice,
        speaker=speaker_hint or profile_name or _resolve_qwen_speaker("default"),
        instruct=str(profile.get("instruct") or "").strip() or None,
        sample_path=str(profile.get("source_sample_path") or "").strip() or None,
        profile_id=str(profile.get("profile_id") or "").strip() or None,
        profile_name=profile_name or None,
        is_profile=True,
    )


def _resolve_qwen_speaker(voice: str) -> str:
    requested = str(voice or "").strip()
    if requested and requested.lower() != "default":
        return requested
    return str(os.getenv("ALPHONSE_QWEN_TTS_SPEAKER") or "Ryan").strip() or "Ryan"


def _resolve_say_fallback_voice(*, requested_voice: str) -> str:
    requested = str(requested_voice or "default").strip() or "default"
    if requested.lower() != "default":
        if resolve_voice_profile(requested) is None:
            return requested
    return str(os.getenv("ALPHONSE_SAY_VOICE") or "default").strip() or "default"


def _with_fallback_metadata(result: dict[str, Any], *, code: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        return result
    output = result.get("output")
    if isinstance(output, dict):
        payload = dict(output)
        payload["fallback_from"] = "qwen"
        payload["fallback_reason_code"] = code
        result["output"] = payload
    return result


def _generate_qwen_custom_voice(
    *,
    model: Any,
    text: str,
    language: str,
    speaker: str,
    instruct: str | None,
    sample_path: str | None,
) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {
        "text": text,
        "language": language,
        "speaker": speaker,
        "instruct": instruct,
    }
    sample = str(sample_path or "").strip()
    if sample:
        sample_file = Path(sample).expanduser().resolve()
        if sample_file.exists() and sample_file.is_file():
            target_key = _resolve_qwen_sample_argument(model)
            if target_key:
                kwargs[target_key] = str(sample_file)
            else:
                logger.info(
                    "local_audio_output_qwen sample_ignored reason=unsupported_param profile_sample_path=%s",
                    str(sample_file),
                )
    return model.generate_custom_voice(**kwargs)


def _resolve_qwen_sample_argument(model: Any) -> str | None:
    fn = getattr(model, "generate_custom_voice", None)
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
    except Exception:
        return None
    names = {name for name in sig.parameters}
    candidates = (
        "reference_audio_path",
        "reference_audio",
        "prompt_audio_path",
        "voice_sample_path",
        "sample_path",
        "audio_prompt_path",
    )
    for key in candidates:
        if key in names:
            return key
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in sig.parameters.values()
    )
    if accepts_kwargs:
        return "reference_audio_path"
    return None


def _first_wav(wavs: Any) -> Any | None:
    if isinstance(wavs, (list, tuple)):
        if not wavs:
            return None
        return wavs[0]
    return wavs


def _resolve_audio_player(file_path: str) -> tuple[list[str], str] | None:
    path = str(file_path or "").strip()
    if not path:
        return None
    if platform.system() == "Darwin":
        return (["afplay", path], "afplay")
    ffplay = shutil.which("ffplay")
    if ffplay:
        return ([ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path], "ffplay")
    return None


def _convert_from_wav(*, source_path: Path, target_format: str) -> dict[str, Any]:
    if target_format == "aiff":
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return _failed(
                "ffmpeg_not_installed",
                "ffmpeg is required to convert audio to aiff for Qwen backend",
                tool="local_audio_output_render",
            )
        target_path = source_path.with_suffix(".aiff")
        converted = subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(source_path), str(target_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if converted.returncode != 0:
            return _failed(
                "audio_convert_failed",
                "failed converting wav to aiff",
                details={"stderr": str(converted.stderr or "").strip()},
                tool="local_audio_output_render",
            )
        source_path.unlink(missing_ok=True)
        return _ok(
            {"file_path": str(target_path), "format": "aiff", "mime_type": "audio/aiff"},
            tool="local_audio_output_render",
        )

    if target_format == "m4a":
        target_path = source_path.with_suffix(".m4a")
        if platform.system() == "Darwin":
            cmd = ["afconvert", "-f", "m4af", "-d", "aac", str(source_path), str(target_path)]
        else:
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                return _failed(
                    "ffmpeg_not_installed",
                    "ffmpeg is required to convert audio to m4a on non-macOS",
                    tool="local_audio_output_render",
                )
            cmd = [ffmpeg_bin, "-y", "-i", str(source_path), "-c:a", "aac", str(target_path)]
        converted = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if converted.returncode != 0:
            return _failed(
                "audio_convert_failed",
                "failed converting wav to m4a",
                details={"stderr": str(converted.stderr or "").strip()},
                tool="local_audio_output_render",
            )
        source_path.unlink(missing_ok=True)
        return _ok(
            {"file_path": str(target_path), "format": "m4a", "mime_type": "audio/mp4"},
            tool="local_audio_output_render",
        )

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return _failed(
            "ffmpeg_not_installed",
            "ffmpeg is required to convert audio to ogg/opus",
            tool="local_audio_output_render",
        )
    target_path = source_path.with_suffix(".ogg")
    converted = subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(source_path), "-c:a", "libopus", "-b:a", "32k", str(target_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if converted.returncode != 0:
        return _failed(
            "audio_convert_failed",
            "failed converting wav to ogg",
            details={"stderr": str(converted.stderr or "").strip()},
            tool="local_audio_output_render",
        )
    source_path.unlink(missing_ok=True)
    return _ok(
        {"file_path": str(target_path), "format": "ogg", "mime_type": "audio/ogg"},
        tool="local_audio_output_render",
    )


def _build_say_command(*, text: str, voice: str, output_path: Path | None = None) -> list[str]:
    cmd = ["say"]
    if voice != "default":
        cmd.extend(["-v", voice])
    if output_path is not None:
        cmd.extend(["-o", str(output_path)])
    cmd.append(text)
    return cmd


def _resolve_render_output_dir(output_dir: str | None) -> Path:
    rendered = str(output_dir or "").strip()
    if rendered:
        return Path(rendered).expanduser().resolve()
    for alias in PRIMARY_WORKDIR_ALIASES:
        record = get_sandbox_alias(alias)
        if not isinstance(record, dict) or not bool(record.get("enabled")):
            continue
        base_path = str(record.get("base_path") or "").strip()
        if base_path:
            return (Path(base_path).expanduser().resolve() / "audio_output").resolve()
    return (default_sandbox_root() / "audio_output").resolve()


def _cleanup_rendered_audio(*, root: Path, keep_path: Path) -> dict[str, int]:
    max_files = _int_env("ALPHONSE_AUDIO_MAX_FILES", default=200, minimum=1)
    max_age_hours = _int_env("ALPHONSE_AUDIO_MAX_AGE_HOURS", default=168, minimum=1)
    cutoff_ts = datetime.now(timezone.utc).timestamp() - float(max_age_hours * 3600)

    removed_age = 0
    candidates = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in {".m4a", ".aiff", ".ogg", ".oga", ".wav"}
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path == keep_path:
            continue
        try:
            if path.stat().st_mtime < cutoff_ts:
                path.unlink(missing_ok=True)
                removed_age += 1
        except Exception:
            continue

    removed_count = 0
    remaining = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in {".m4a", ".aiff", ".ogg", ".oga", ".wav"}
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in remaining[max_files:]:
        if path == keep_path:
            continue
        try:
            path.unlink(missing_ok=True)
            removed_count += 1
        except Exception:
            continue

    return {
        "max_files": max_files,
        "max_age_hours": max_age_hours,
        "removed_by_age": removed_age,
        "removed_by_count": removed_count,
    }


def _int_env(name: str, *, default: int, minimum: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


def _watch_say_process(proc: subprocess.Popen[str], preview: str) -> None:
    return_code = proc.wait()
    stderr = ""
    if proc.stderr is not None:
        try:
            stderr = str(proc.stderr.read() or "").strip()
        except Exception:
            stderr = ""
    if return_code != 0:
        logger.error(
            "local_audio_output_speak async_failed backend=say pid=%s returncode=%s stderr=%s preview=%s",
            proc.pid,
            return_code,
            stderr,
            preview,
        )
        return
    logger.info("local_audio_output_speak async_done backend=say pid=%s preview=%s", proc.pid, preview)


def _ok(result: dict[str, Any], *, tool: str = "local_audio_output_speak") -> dict[str, Any]:
    return {
        "output": result,
        "exception": None,
        "metadata": {"tool": tool},
    }


def _failed(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    *,
    tool: str = "local_audio_output_speak",
) -> dict[str, Any]:
    return {
        "output": None,
        "exception": {
            "code": str(code),
            "message": str(message),
            "retryable": False,
            "details": dict(details or {}),
        },
        "metadata": {"tool": tool},
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m alphonse.agent.tools.local_audio_output")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--volume", type=float, default=None)
    args = parser.parse_args()

    tool = LocalAudioOutputSpeakTool()
    result = tool.execute(
        text=args.text,
        voice=args.voice,
        blocking=bool(args.blocking),
        volume=args.volume,
    )
    print(result)


if __name__ == "__main__":
    main()
