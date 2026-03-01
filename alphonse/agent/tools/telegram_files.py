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
            return _failed("telegram_get_file_meta", "telegram_bot_token_missing")
        meta = _TelegramFileClient(self._bot_token).get_file(file_id=file_id)
        return _ok(
            "telegram_get_file_meta",
            {
                "file_id": file_id,
                "file_path": meta.get("file_path"),
                "file_size": meta.get("file_size"),
                "file_unique_id": meta.get("file_unique_id"),
            },
        )


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
            return _failed("telegram_download_file", "telegram_bot_token_missing")
        try:
            result = _TelegramFileClient(self._bot_token).download_file(
                file_id=file_id,
                sandbox_alias=str(sandbox_alias or DEFAULT_SANDBOX_ALIAS),
                relative_path=relative_path,
            )
        except Exception as exc:
            return _failed("telegram_download_file", str(exc) or "download_failed")
        return _ok("telegram_download_file", result)


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
            return _failed("transcribe_telegram_audio", "telegram_bot_token_missing")
        download = TelegramDownloadFileTool(bot_token=self._bot_token).execute(
            file_id=file_id,
            sandbox_alias=sandbox_alias,
        )
        if str(download.get("status") or "") != "ok":
            return download
        payload = download.get("result") if isinstance(download.get("result"), dict) else {}
        relative_path = str(payload.get("relative_path") or "").strip()
        alias = str(payload.get("sandbox_alias") or sandbox_alias or DEFAULT_SANDBOX_ALIAS).strip()
        if not relative_path:
            return _failed("transcribe_telegram_audio", "audio_path_missing", retryable=False)
        local_path = resolve_sandbox_path(alias=alias, relative_path=relative_path)
        whisper_cmd = shutil.which("whisper")
        if not whisper_cmd:
                return _failed(
                    "transcribe_telegram_audio",
                    "whisper_cli_not_found",
                    retryable=False,
                    details={"sandbox_alias": alias, "relative_path": relative_path},
                )
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
                    return _failed(
                        "transcribe_telegram_audio",
                        "whisper_transcription_failed",
                        retryable=True,
                        details={"sandbox_alias": alias, "relative_path": relative_path},
                    )
                payload = _read_whisper_output(Path(tmp_dir))
            text = payload.get("text") if isinstance(payload, dict) else None
            segments = payload.get("segments") if isinstance(payload, dict) else None
            normalized_segments = _normalize_segments(segments)
            if not str(text or "").strip():
                return _failed(
                    "transcribe_telegram_audio",
                    "transcript_empty",
                    retryable=True,
                    details={"sandbox_alias": alias, "relative_path": relative_path},
                )
            return _ok(
                "transcribe_telegram_audio",
                {
                    "file_id": file_id,
                    "sandbox_alias": alias,
                    "relative_path": relative_path,
                    "text": str(text or ""),
                    "segments": normalized_segments,
                    "model": self._model,
                },
            )
        except Exception as exc:
            return _failed("transcribe_telegram_audio", str(exc) or "transcription_failed", retryable=True)


class VisionAnalyzeImageTool:
    def __init__(self) -> None:
        self._ollama_base_url = str(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
        self._model = _resolve_vision_model(
            primary_env="ALPHONSE_VISION_ANALYZE_MODEL",
            default_model="qwen3-vl:4b",
        )
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
            return _failed("vision_analyze_image", "image_path_missing")
        local_path = resolve_sandbox_path(alias=alias, relative_path=rel)
        if not local_path.exists():
            return _failed("vision_analyze_image", "image_not_found")
        if not local_path.is_file():
            return _failed("vision_analyze_image", "image_not_a_file")
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
                return _failed_http("vision_analyze_image", resp)
            body = resp.json() if resp.content else {}
            text = _extract_message_content_text(body)
            return _ok(
                "vision_analyze_image",
                {
                    "sandbox_alias": alias,
                    "relative_path": rel,
                    "analysis": text,
                    "model": self._model,
                },
            )
        except Exception as exc:
            return _failed("vision_analyze_image", str(exc) or "vision_failed")


class VisionExtractTool:
    def __init__(self) -> None:
        self._ollama_base_url = str(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
        self._model = _resolve_vision_model(
            primary_env="ALPHONSE_VISION_EXTRACT_MODEL",
            default_model="qwen3-vl:4b",
        )
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
            return _failed("vision_extract", "image_path_missing")
        local_path = resolve_sandbox_path(alias=alias, relative_path=rel)
        if not local_path.exists():
            return _failed("vision_extract", "image_not_found")
        if not local_path.is_file():
            return _failed("vision_extract", "image_not_a_file")
        try:
            image_bytes = local_path.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            user_text = str(
                prompt
                or (
                    "Extract all visible text from this image exactly as written. "
                    "Preserve line breaks when possible. "
                    "Do not infer or hallucinate missing text."
                )
            )
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
                return _failed_http("vision_extract", resp)
            body = resp.json() if resp.content else {}
            text, blocks = _extract_ocr_text_and_blocks(body)
            return _ok(
                "vision_extract",
                {
                    "sandbox_alias": alias,
                    "relative_path": rel,
                    "text": text,
                    "blocks": blocks,
                    "model": self._model,
                },
            )
        except Exception as exc:
            return _failed("vision_extract", str(exc) or "vision_failed")


def _read_whisper_output(output_dir: Path) -> dict[str, Any] | None:
    files = sorted(output_dir.glob("*.json"))
    if not files:
        return None
    try:
        parsed = json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_vision_model(*, primary_env: str, default_model: str) -> str:
    return str(
        os.getenv(primary_env)
        or os.getenv("ALPHONSE_VISION_MODEL")
        or default_model
    ).strip() or default_model


def _extract_message_content_text(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    message = body.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _extract_ocr_text_and_blocks(body: Any) -> tuple[str, list[dict[str, Any]]]:
    raw_text = _extract_message_content_text(body).strip()
    parsed: Any = None
    if raw_text:
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        text = str(parsed.get("text") or "").strip()
        if text:
            blocks = _normalize_ocr_blocks(parsed.get("blocks"), fallback_text=text)
            return text, blocks
    blocks = _normalize_ocr_blocks(None, fallback_text=raw_text)
    return raw_text, blocks


def _normalize_ocr_blocks(raw_blocks: Any, *, fallback_text: str) -> list[dict[str, Any]]:
    if isinstance(raw_blocks, list):
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_blocks):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            block: dict[str, Any] = {"text": text}
            kind = item.get("kind")
            if kind is not None:
                block["kind"] = str(kind)
            index_value = item.get("index")
            if isinstance(index_value, int):
                block["index"] = index_value
            else:
                block["index"] = idx
            bbox = item.get("bbox")
            if isinstance(bbox, dict):
                block["bbox"] = bbox
            out.append(block)
        if out:
            return out
    fallback = fallback_text.strip()
    if not fallback:
        return []
    return [{"text": fallback, "kind": "paragraph", "index": 0}]


def _language_code(language_hint: str | None) -> str | None:
    hint = str(language_hint or "").strip().lower()
    if not hint:
        return None
    if "-" in hint:
        hint = hint.split("-", 1)[0]
    if "_" in hint:
        hint = hint.split("_", 1)[0]
    return hint or None


def _ok(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "result": result,
        "error": None,
        "metadata": {"tool": tool},
    }


def _failed(
    tool: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {
            "code": message,
            "message": message,
            "retryable": bool(retryable),
            "details": dict(details or {}),
        },
        "metadata": {"tool": tool},
    }


def _failed_http(tool: str, response: requests.Response) -> dict[str, Any]:
    status_code = int(getattr(response, "status_code", 0) or 0)
    code = f"vision_http_{status_code}"
    preview = _http_response_preview(response)
    message = f"{code}: {preview}" if preview else code
    details: dict[str, Any] = {"status_code": status_code}
    if preview:
        details["response_preview"] = preview
    return {
        "status": "failed",
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "metadata": {"tool": tool},
    }


def _http_response_preview(response: requests.Response) -> str:
    text = ""
    try:
        raw_text = getattr(response, "text", None)
        if isinstance(raw_text, str) and raw_text.strip():
            text = raw_text.strip()
        else:
            raw_content = getattr(response, "content", b"")
            if isinstance(raw_content, bytes):
                text = raw_content.decode("utf-8", errors="replace").strip()
            elif raw_content:
                text = str(raw_content).strip()
    except Exception:
        text = ""
    if not text:
        try:
            body = response.json()
            if body is not None:
                text = json.dumps(body, ensure_ascii=False)
        except Exception:
            text = ""
    text = " ".join(str(text or "").split())
    if len(text) > 300:
        return text[:297].rstrip() + "..."
    return text


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
