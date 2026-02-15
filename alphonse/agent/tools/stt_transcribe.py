from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from alphonse.agent.nervous_system.assets import get_asset
from alphonse.agent.nervous_system.assets import resolve_asset_path


class SttTranscribeTool:
    def __init__(self, *, model: str | None = None) -> None:
        self._model = str(model or os.getenv("ALPHONSE_STT_MODEL") or "base").strip() or "base"

    def execute(self, *, asset_id: str, language_hint: str | None = None) -> dict[str, Any]:
        normalized_asset_id = str(asset_id or "").strip()
        if not normalized_asset_id:
            return _failed("asset_id_required", retryable=False, asset_id=normalized_asset_id)

        asset = get_asset(normalized_asset_id)
        if not isinstance(asset, dict):
            return _failed("asset_not_found", retryable=False, asset_id=normalized_asset_id)
        if str(asset.get("kind") or "").strip().lower() != "audio":
            return _failed("asset_not_audio", retryable=False, asset_id=normalized_asset_id)

        whisper_cmd = shutil.which("whisper")
        if not whisper_cmd:
            return _failed("whisper_cli_not_found", retryable=False, asset_id=normalized_asset_id)

        try:
            audio_path = resolve_asset_path(normalized_asset_id)
        except Exception:
            return _failed("asset_path_unavailable", retryable=True, asset_id=normalized_asset_id)
        if not audio_path.exists():
            return _failed("asset_file_missing", retryable=True, asset_id=normalized_asset_id)

        language = _language_code(language_hint)
        with tempfile.TemporaryDirectory(prefix="alphonse-stt-") as tmp_dir:
            cmd = [
                whisper_cmd,
                str(audio_path),
                "--model",
                self._model,
                "--output_dir",
                tmp_dir,
                "--output_format",
                "json",
            ]
            if language:
                cmd.extend(["--language", language])
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                return _failed("whisper_transcription_failed", retryable=True, asset_id=normalized_asset_id)

            payload = _read_whisper_output(Path(tmp_dir))
            if not isinstance(payload, dict):
                return _failed("whisper_output_missing", retryable=True, asset_id=normalized_asset_id)
            text = str(payload.get("text") or "").strip()
            segments = _normalize_segments(payload.get("segments"))
            if not text:
                return _failed("transcript_empty", retryable=True, asset_id=normalized_asset_id)
            return {
                "status": "ok",
                "asset_id": normalized_asset_id,
                "text": text,
                "segments": segments,
            }


def _failed(error: str, *, retryable: bool, asset_id: str | None) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": str(error or "stt_transcribe_failed"),
        "retryable": bool(retryable),
        "asset_id": str(asset_id or "").strip() or None,
    }


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
