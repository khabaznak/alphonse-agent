from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from alphonse.agent.observability.log_manager import get_component_logger
import platform
import subprocess
import threading
from typing import Any

from alphonse.agent.nervous_system.sandbox_dirs import (
    PRIMARY_WORKDIR_ALIASES,
    default_sandbox_root,
    get_sandbox_alias,
)

logger = get_component_logger("tools.local_audio_output")


class LocalAudioOutputSpeakTool:
    """Local audio output POC using macOS native TTS.

    Example tool-call JSON:
    {
      "tool": "local_audio_output.speak",
      "args": { "text": "Hello World", "blocking": false }
    }
    """

    def execute(
        self,
        *,
        text: str,
        voice: str = "default",
        blocking: bool = False,
        volume: float | None = None,
    ) -> dict[str, Any]:
        _ = volume
        spoken_text = str(text or "").strip()
        selected_voice = str(voice or "default").strip() or "default"
        is_blocking = bool(blocking)
        preview = spoken_text[:40] + ("..." if len(spoken_text) > 40 else "")

        logger.info(
            "local_audio_output.speak invoke text_len=%s voice=%s blocking=%s preview=%s",
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
                "local_audio_output.speak is currently supported only on macOS.",
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
                    "local_audio_output.speak failed returncode=%s stderr=%s",
                    completed.returncode,
                    stderr,
                )
                return {
                    "status": "failed",
                    "result": None,
                    "error": {
                        "code": "tts_command_failed",
                        "message": "tts command failed",
                        "retryable": False,
                        "details": {"stderr": stderr},
                    },
                    "metadata": {"tool": "local_audio_output.speak"},
                }
            logger.info("local_audio_output.speak success mode=blocking")
            return _ok({"mode": "blocking"})

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
        logger.info("local_audio_output.speak success mode=non_blocking pid=%s", proc.pid)
        return _ok({"mode": "non_blocking", "pid": int(proc.pid)})


class LocalAudioOutputRenderTool:
    """Render text-to-speech to an audio file for downstream integrations."""

    def execute(
        self,
        *,
        text: str,
        voice: str = "default",
        output_dir: str | None = None,
        filename_prefix: str = "response",
        format: str = "m4a",
    ) -> dict[str, Any]:
        spoken_text = str(text or "").strip()
        if not spoken_text:
            return _failed("text_required", "text is required", tool="local_audio_output.render")
        if platform.system() != "Darwin":
            return _failed(
                "local_audio_output_not_supported",
                "local_audio_output.render is currently supported only on macOS.",
                tool="local_audio_output.render",
            )
        selected_voice = str(voice or "default").strip() or "default"
        target_format = str(format or "m4a").strip().lower()
        if target_format not in {"aiff", "m4a", "ogg"}:
            return _failed(
                "unsupported_audio_format",
                "Supported formats are: aiff, m4a, ogg",
                tool="local_audio_output.render",
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
                tool="local_audio_output.render",
            )
        if target_format == "aiff":
            cleanup = _cleanup_rendered_audio(root=root, keep_path=source_path)
            return _ok(
                {
                    "file_path": str(source_path),
                    "format": "aiff",
                    "mime_type": "audio/aiff",
                    "retention": cleanup,
                },
                tool="local_audio_output.render",
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
                    tool="local_audio_output.render",
                )
            mime_type = "audio/mp4"
        else:
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                return _failed(
                    "ffmpeg_not_installed",
                    "ffmpeg is required to convert audio to ogg/opus",
                    tool="local_audio_output.render",
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
                    tool="local_audio_output.render",
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
            },
            tool="local_audio_output.render",
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
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".m4a", ".aiff", ".ogg", ".oga"}],
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
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".m4a", ".aiff", ".ogg", ".oga"}],
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
            "local_audio_output.speak async_failed pid=%s returncode=%s stderr=%s preview=%s",
            proc.pid,
            return_code,
            stderr,
            preview,
        )
        return
    logger.info("local_audio_output.speak async_done pid=%s preview=%s", proc.pid, preview)


def _ok(result: dict[str, Any], *, tool: str = "local_audio_output.speak") -> dict[str, Any]:
    return {
        "status": "ok",
        "result": result,
        "error": None,
        "metadata": {"tool": tool},
    }


def _failed(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    *,
    tool: str = "local_audio_output.speak",
) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {
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
