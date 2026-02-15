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
            return {"status": "failed", "error": "asset_id_required"}

        asset = get_asset(normalized_asset_id)
        if not isinstance(asset, dict):
            return {"status": "failed", "error": "asset_not_found"}
        if str(asset.get("kind") or "").strip().lower() != "audio":
            return {"status": "failed", "error": "asset_not_audio"}

        whisper_cmd = shutil.which("whisper")
        if not whisper_cmd:
            return {"status": "failed", "error": "whisper_cli_not_found"}

        try:
            audio_path = resolve_asset_path(normalized_asset_id)
        except Exception:
            return {"status": "failed", "error": "asset_path_unavailable"}
        if not audio_path.exists():
            return {"status": "failed", "error": "asset_file_missing"}

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
                return {"status": "failed", "error": "whisper_transcription_failed"}

            payload = _read_whisper_output(Path(tmp_dir))
            if not isinstance(payload, dict):
                return {"status": "failed", "error": "whisper_output_missing"}
            text = str(payload.get("text") or "").strip()
            segments = _normalize_segments(payload.get("segments"))
            if not text:
                return {"status": "failed", "error": "transcript_empty"}
            return {
                "status": "ok",
                "asset_id": normalized_asset_id,
                "text": text,
                "segments": segments,
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
