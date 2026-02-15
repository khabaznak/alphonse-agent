from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter


def _default_download_dir() -> Path:
    root = Path(os.getenv("ALPHONSE_TELEGRAM_FILES_DIR") or "/tmp/alphonse-telegram-files")
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class _TelegramFileClient:
    bot_token: str

    def _adapter(self) -> TelegramAdapter:
        return TelegramAdapter({"bot_token": self.bot_token, "poll_interval_sec": 1.0})

    def get_file(self, file_id: str) -> dict[str, Any]:
        return self._adapter().get_file(file_id=file_id)

    def download_file(self, *, file_id: str, destination_dir: Path) -> dict[str, Any]:
        meta = self.get_file(file_id)
        file_path = str(meta.get("file_path") or "").strip()
        if not file_path:
            raise RuntimeError("telegram_file_path_missing")
        payload = self._adapter().download_file(file_path=file_path)
        filename = Path(file_path).name or f"{file_id}.bin"
        local_path = destination_dir / filename
        local_path.write_bytes(payload)
        return {
            "file_id": file_id,
            "file_path": file_path,
            "file_size": int(meta.get("file_size") or len(payload)),
            "local_path": str(local_path),
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

    def execute(self, *, file_id: str) -> dict[str, Any]:
        if not self._bot_token:
            return {"status": "failed", "error": "telegram_bot_token_missing"}
        try:
            result = _TelegramFileClient(self._bot_token).download_file(
                file_id=file_id,
                destination_dir=_default_download_dir(),
            )
        except Exception as exc:
            return {"status": "failed", "error": str(exc) or "download_failed"}
        result["status"] = "ok"
        return result


class TranscribeTelegramAudioTool:
    def __init__(self, *, bot_token: str | None = None) -> None:
        self._bot_token = str(bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        self._openai_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
        self._model = str(os.getenv("OPENAI_AUDIO_MODEL") or "gpt-4o-mini-transcribe")

    def execute(self, *, file_id: str, language: str | None = None) -> dict[str, Any]:
        if not self._bot_token:
            return {"status": "failed", "error": "telegram_bot_token_missing"}
        download = TelegramDownloadFileTool(bot_token=self._bot_token).execute(file_id=file_id)
        if download.get("status") != "ok":
            return download
        if not self._openai_key:
            return {
                "status": "failed",
                "error": "openai_api_key_missing",
                "local_path": download.get("local_path"),
            }
        local_path = str(download.get("local_path") or "").strip()
        if not local_path:
            return {"status": "failed", "error": "audio_path_missing"}
        try:
            with open(local_path, "rb") as fh:
                files = {"file": (Path(local_path).name, fh, download.get("mime_type") or "audio/ogg")}
                data: dict[str, Any] = {"model": self._model}
                if isinstance(language, str) and language.strip():
                    data["language"] = language.strip()
                resp = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._openai_key}"},
                    files=files,
                    data=data,
                    timeout=60,
                )
            if resp.status_code >= 400:
                return {"status": "failed", "error": f"transcription_http_{resp.status_code}"}
            payload = resp.json() if resp.content else {}
            text = payload.get("text") if isinstance(payload, dict) else None
            return {
                "status": "ok",
                "file_id": file_id,
                "local_path": local_path,
                "text": str(text or ""),
                "model": self._model,
            }
        except Exception as exc:
            return {"status": "failed", "error": str(exc) or "transcription_failed"}


class AnalyzeTelegramImageTool:
    def __init__(self, *, bot_token: str | None = None) -> None:
        self._bot_token = str(bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        self._openai_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
        self._model = str(os.getenv("OPENAI_VISION_MODEL") or "gpt-4o-mini")

    def execute(self, *, file_id: str, prompt: str | None = None) -> dict[str, Any]:
        if not self._bot_token:
            return {"status": "failed", "error": "telegram_bot_token_missing"}
        download = TelegramDownloadFileTool(bot_token=self._bot_token).execute(file_id=file_id)
        if download.get("status") != "ok":
            return download
        if not self._openai_key:
            return {
                "status": "failed",
                "error": "openai_api_key_missing",
                "local_path": download.get("local_path"),
            }
        local_path = str(download.get("local_path") or "").strip()
        if not local_path:
            return {"status": "failed", "error": "image_path_missing"}
        try:
            image_bytes = Path(local_path).read_bytes()
            mime = str(download.get("mime_type") or "image/jpeg")
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            user_text = str(prompt or "Describe this image concisely.")
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                            },
                        ],
                    }
                ],
            }
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openai_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            if resp.status_code >= 400:
                return {"status": "failed", "error": f"vision_http_{resp.status_code}"}
            body = resp.json() if resp.content else {}
            text = ""
            if isinstance(body, dict):
                choices = body.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0] if isinstance(choices[0], dict) else {}
                    msg = first.get("message") if isinstance(first, dict) else {}
                    if isinstance(msg, dict):
                        text = str(msg.get("content") or "")
            return {
                "status": "ok",
                "file_id": file_id,
                "local_path": local_path,
                "analysis": text,
                "model": self._model,
            }
        except Exception as exc:
            return {"status": "failed", "error": str(exc) or "vision_failed"}
