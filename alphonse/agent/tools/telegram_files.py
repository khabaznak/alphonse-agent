from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.agent.nervous_system.sandbox_dirs import DEFAULT_SANDBOX_ALIAS
from alphonse.agent.nervous_system.sandbox_dirs import resolve_sandbox_path


@dataclass
class _TelegramFileClient:
    bot_token: str

    def _adapter(self) -> TelegramAdapter:
        return TelegramAdapter({"bot_token": self.bot_token, "poll_interval_sec": 1.0})

    def get_file(self, file_id: str) -> dict[str, Any]:
        return self._adapter().get_file(file_id=file_id)

    def download_file(
        self,
        *,
        file_id: str,
        sandbox_alias: str,
        relative_path: str | None = None,
    ) -> dict[str, Any]:
        meta = self.get_file(file_id)
        file_path = str(meta.get("file_path") or "").strip()
        if not file_path:
            raise RuntimeError("telegram_file_path_missing")
        payload = self._adapter().download_file(file_path=file_path)
        filename = Path(file_path).name or f"{file_id}.bin"
        chosen_relative = str(relative_path or filename).strip() or filename
        local_path = resolve_sandbox_path(alias=sandbox_alias, relative_path=chosen_relative)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(payload)
        return {
            "file_id": file_id,
            "file_path": file_path,
            "file_size": int(meta.get("file_size") or len(payload)),
            "sandbox_alias": sandbox_alias,
            "relative_path": chosen_relative,
            "mime_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
        }


class TelegramGetFileMetaTool:
    def __init__(self, *, bot_token: str | None = None) -> None:
        self._bot_token = str(bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()

    def execute(self, *, file_id: str) -> dict[str, Any]:
        if not self._bot_token:
            return {"status": "failed", "error": "telegram_bot_token_missing"}
        meta = _TelegramFileClient(self._bot_token).get_file(file_id=file_id)
        return {
            "status": "ok",
            "file_id": file_id,
            "file_path": meta.get("file_path"),
            "file_size": meta.get("file_size"),
            "file_unique_id": meta.get("file_unique_id"),
        }


class TelegramDownloadFileTool:
    def __init__(self, *, bot_token: str | None = None) -> None:
        self._bot_token = str(bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()

    def execute(
        self,
        *,
        file_id: str,
        sandbox_alias: str = DEFAULT_SANDBOX_ALIAS,
        relative_path: str | None = None,
    ) -> dict[str, Any]:
        if not self._bot_token:
            return {"status": "failed", "error": "telegram_bot_token_missing"}
        try:
            result = _TelegramFileClient(self._bot_token).download_file(
                file_id=file_id,
                sandbox_alias=str(sandbox_alias or DEFAULT_SANDBOX_ALIAS),
                relative_path=relative_path,
            )
        except Exception as exc:
            return {"status": "failed", "error": str(exc) or "download_failed"}
        result["status"] = "ok"
        return result


class TranscribeTelegramAudioTool:
    def __init__(self, *, bot_token: str | None = None) -> None:
        self._bot_token = str(bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        self._model = str(os.getenv("ALPHONSE_STT_MODEL") or "base").strip() or "base"

    def execute(
        self,
        *,
        file_id: str,
        language: str | None = None,
        sandbox_alias: str = DEFAULT_SANDBOX_ALIAS,
    ) -> dict[str, Any]:
        if not self._bot_token:
            return {"status": "failed", "error": "telegram_bot_token_missing"}
        download = TelegramDownloadFileTool(bot_token=self._bot_token).execute(
            file_id=file_id,
            sandbox_alias=sandbox_alias,
        )
        if download.get("status") != "ok":
            return download
        relative_path = str(download.get("relative_path") or "").strip()
        alias = str(download.get("sandbox_alias") or sandbox_alias or DEFAULT_SANDBOX_ALIAS).strip()
        if not relative_path:
            return {"status": "failed", "error": "audio_path_missing", "retryable": False}
        local_path = resolve_sandbox_path(alias=alias, relative_path=relative_path)
        whisper_cmd = shutil.which("whisper")
        if not whisper_cmd:
            return {
                "status": "failed",
                "error": "whisper_cli_not_found",
                "retryable": False,
                "sandbox_alias": alias,
                "relative_path": relative_path,
            }
        try:
            language_code = _language_code(language)
            with tempfile.TemporaryDirectory(prefix="alphonse-telegram-stt-") as tmp_dir:
                cmd = [
                    whisper_cmd,
                    str(local_path),
                    "--model",
                    self._model,
                    "--output_dir",
                    tmp_dir,
                    "--output_format",
                    "json",
                ]
                if language_code:
                    cmd.extend(["--language", language_code])
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    return {
                        "status": "failed",
                        "error": "whisper_transcription_failed",
                        "retryable": True,
                        "sandbox_alias": alias,
                        "relative_path": relative_path,
                    }
                payload = _read_whisper_output(Path(tmp_dir))
            text = payload.get("text") if isinstance(payload, dict) else None
            segments = payload.get("segments") if isinstance(payload, dict) else None
            normalized_segments = _normalize_segments(segments)
            if not str(text or "").strip():
                return {
                    "status": "failed",
                    "error": "transcript_empty",
                    "retryable": True,
                    "sandbox_alias": alias,
                    "relative_path": relative_path,
                }
            return {
                "status": "ok",
                "file_id": file_id,
                "sandbox_alias": alias,
                "relative_path": relative_path,
                "text": str(text or ""),
                "segments": normalized_segments,
                "model": self._model,
            }
        except Exception as exc:
            return {"status": "failed", "error": str(exc) or "transcription_failed", "retryable": True}


class AnalyzeTelegramImageTool:
    def __init__(self, *, bot_token: str | None = None) -> None:
        self._bot_token = str(bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        self._vision = VisionAnalyzeImageTool()

    def execute(
        self,
        *,
        file_id: str | None = None,
        prompt: str | None = None,
        sandbox_alias: str = DEFAULT_SANDBOX_ALIAS,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._bot_token:
            return {"status": "failed", "error": "telegram_bot_token_missing"}
        selected_file_id = str(file_id or "").strip() or _extract_image_file_id_from_state(state)
        if not selected_file_id:
            return {"status": "failed", "error": "telegram_image_file_id_missing"}
        user_scope = _state_user_scope(state)
        scoped_relative_path = _scoped_telegram_relative_path(
            file_id=selected_file_id,
            user_scope=user_scope,
        )
        download = TelegramDownloadFileTool(bot_token=self._bot_token).execute(
            file_id=selected_file_id,
            sandbox_alias=sandbox_alias,
            relative_path=scoped_relative_path,
        )
        if download.get("status") != "ok":
            return download
        relative_path = str(download.get("relative_path") or "").strip()
        alias = str(download.get("sandbox_alias") or sandbox_alias or DEFAULT_SANDBOX_ALIAS).strip()
        if not relative_path:
            return {"status": "failed", "error": "image_path_missing"}
        result = self._vision.execute(
            sandbox_alias=alias,
            relative_path=relative_path,
            prompt=prompt,
        )
        if result.get("status") != "ok":
            return result
        return {
            "status": "ok",
            "file_id": selected_file_id,
            "sandbox_alias": alias,
            "relative_path": relative_path,
            "analysis": result.get("analysis"),
            "model": result.get("model"),
        }


class VisionAnalyzeImageTool:
    def __init__(self) -> None:
        self._ollama_base_url = str(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
        self._model = str(os.getenv("ALPHONSE_VISION_MODEL") or "qwen3-vl:4b").strip() or "qwen3-vl:4b"
        self._timeout_seconds = int(str(os.getenv("ALPHONSE_VISION_TIMEOUT_SECONDS") or "60").strip() or "60")

    def execute(
        self,
        *,
        sandbox_alias: str,
        relative_path: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        alias = str(sandbox_alias or DEFAULT_SANDBOX_ALIAS).strip() or DEFAULT_SANDBOX_ALIAS
        rel = str(relative_path or "").strip()
        if not rel:
            return {"status": "failed", "error": "image_path_missing"}
        local_path = resolve_sandbox_path(alias=alias, relative_path=rel)
        if not local_path.exists():
            return {"status": "failed", "error": "image_not_found"}
        if not local_path.is_file():
            return {"status": "failed", "error": "image_not_a_file"}
        try:
            image_bytes = local_path.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            user_text = str(prompt or "Describe this image concisely in the user's language.")
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": user_text,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
            }
            base_url = self._ollama_base_url.rstrip("/")
            resp = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=self._timeout_seconds,
            )
            if resp.status_code >= 400:
                return {"status": "failed", "error": f"vision_http_{resp.status_code}"}
            body = resp.json() if resp.content else {}
            text = ""
            if isinstance(body, dict):
                message = body.get("message")
                if isinstance(message, dict):
                    text = str(message.get("content") or "")
            return {
                "status": "ok",
                "sandbox_alias": alias,
                "relative_path": rel,
                "analysis": text,
                "model": self._model,
            }
        except Exception as exc:
            return {"status": "failed", "error": str(exc) or "vision_failed"}


def _extract_image_file_id_from_state(state: dict[str, Any] | None) -> str:
    if not isinstance(state, dict):
        return ""
    raw = state.get("provider_event")
    if not isinstance(raw, dict):
        return ""
    message = raw.get("message")
    if not isinstance(message, dict):
        return ""
    photo = message.get("photo")
    if isinstance(photo, list):
        for item in reversed(photo):
            if not isinstance(item, dict):
                continue
            file_id = str(item.get("file_id") or "").strip()
            if file_id:
                return file_id
    document = message.get("document")
    if isinstance(document, dict):
        mime = str(document.get("mime_type") or "").strip().lower()
        file_id = str(document.get("file_id") or "").strip()
    if file_id and mime.startswith("image/"):
        return file_id
    return ""


def _state_user_scope(state: dict[str, Any] | None) -> str:
    if not isinstance(state, dict):
        return "unknown"
    for key in ("user_id", "channel_target", "session_id"):
        value = str(state.get(key) or "").strip()
        if value:
            return _sanitize_path_segment(value)
    return "unknown"


def _sanitize_path_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _scoped_telegram_relative_path(*, file_id: str, user_scope: str) -> str:
    safe_file_id = _sanitize_path_segment(file_id)
    return f"users/{user_scope}/images/{safe_file_id}.bin"


def _read_whisper_output(output_dir: Path) -> dict[str, Any] | None:
    files = sorted(output_dir.glob("*.json"))
    if not files:
        return None
    try:
        parsed = json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _language_code(language_hint: str | None) -> str | None:
    hint = str(language_hint or "").strip().lower()
    if not hint:
        return None
    if "-" in hint:
        hint = hint.split("-", 1)[0]
    if "_" in hint:
        hint = hint.split("_", 1)[0]
    return hint or None


def _normalize_segments(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:200]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "start": item.get("start"),
                "end": item.get("end"),
                "text": text,
            }
        )
    return out
