from __future__ import annotations

import argparse
import logging
import platform
import subprocess
import threading
from typing import Any

logger = logging.getLogger(__name__)


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


def _build_say_command(*, text: str, voice: str) -> list[str]:
    cmd = ["say"]
    if voice != "default":
        cmd.extend(["-v", voice])
    cmd.append(text)
    return cmd


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


def _ok(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "result": result,
        "error": None,
        "metadata": {"tool": "local_audio_output.speak"},
    }


def _failed(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {
            "code": str(code),
            "message": str(message),
            "retryable": False,
            "details": dict(details or {}),
        },
        "metadata": {"tool": "local_audio_output.speak"},
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
