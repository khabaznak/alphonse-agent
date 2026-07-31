"""Persistent configuration and readiness for optional local media backends."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from alphonse.agent_v2.database import connect_database, default_database_path


@dataclass(frozen=True)
class VerificationState:
    ready: bool = False
    verified_at: str = ""
    error: str = ""
    preview: str = ""


@dataclass(frozen=True)
class TtsSettings:
    enabled: bool = False
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    device_map: str = "auto"
    dtype: str = ""
    language: str = "Auto"
    speaker: str = "Ryan"
    instruct: str = ""
    attn_implementation: str = ""
    local_files_only: bool = True
    verification: VerificationState = VerificationState()

    @property
    def available(self) -> bool: return self.enabled and self.verification.ready


@dataclass(frozen=True)
class SttSettings:
    enabled: bool = False
    executable_path: str = ""
    model: str = "base"
    default_language: str = ""
    verification: VerificationState = VerificationState()

    @property
    def available(self) -> bool: return self.enabled and self.verification.ready


@dataclass(frozen=True)
class OcrSettings:
    enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    model_id: str = "qwen3-vl:4b"
    timeout_seconds: float = 60.0
    verification: VerificationState = VerificationState()

    @property
    def available(self) -> bool: return self.enabled and self.verification.ready


@dataclass(frozen=True)
class MediaToolsSettings:
    tts: TtsSettings = TtsSettings()
    stt: SttSettings = SttSettings()
    ocr: OcrSettings = OcrSettings()

    def to_dict(self) -> dict[str, object]:
        def render(value: object) -> dict[str, object]:
            body = asdict(value)
            verification = body.pop("verification", {})
            body["verification"] = verification
            body["available"] = bool(getattr(value, "available"))
            return body
        return {"tts": render(self.tts), "stt": render(self.stt), "ocr": render(self.ocr), "platform": _platform_name(), "say_available": _say_available()}


class SQLiteMediaToolsSettingsStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None: self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteMediaToolsSettingsStore":
        return cls(default_database_path())

    def get(self) -> MediaToolsSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM v2_media_tools_settings WHERE settings_id=1").fetchone()
        if row is None: return MediaToolsSettings()
        import json
        try: return _from_dict(json.loads(str(row["payload"])))
        except Exception: return MediaToolsSettings()

    def save(self, settings: MediaToolsSettings) -> MediaToolsSettings:
        import json
        normalized = _validate(settings)
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO v2_media_tools_settings(settings_id,payload,updated_at) VALUES (1,?,?)", (json.dumps(_to_storage(normalized), ensure_ascii=False), _now()))
        return self.get()

    def update(self, kind: str, values: dict[str, object]) -> MediaToolsSettings:
        current = self.get()
        if kind == "tts": updated = replace(current, tts=_tts_from_values(current.tts, values))
        elif kind == "stt": updated = replace(current, stt=_stt_from_values(current.stt, values))
        elif kind == "ocr": updated = replace(current, ocr=_ocr_from_values(current.ocr, values))
        else: raise ValueError("media_tools_kind_invalid")
        return self.save(updated)

    def mark_verification(self, kind: str, *, ready: bool, error: str = "", preview: str = "") -> MediaToolsSettings:
        current = self.get(); state = VerificationState(ready=ready, verified_at=_now(), error=str(error or "")[:1000], preview=str(preview or "")[:2000])
        if kind == "tts": updated = replace(current, tts=replace(current.tts, verification=state))
        elif kind == "stt": updated = replace(current, stt=replace(current.stt, verification=state))
        elif kind == "ocr": updated = replace(current, ocr=replace(current.ocr, verification=state))
        else: raise ValueError("media_tools_kind_invalid")
        return self.save(updated)

    def _connect(self):
        if self._memory is not None: return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS v2_media_tools_settings (settings_id INTEGER PRIMARY KEY CHECK(settings_id=1), payload TEXT NOT NULL, updated_at TEXT NOT NULL) STRICT")


class _Connection:
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection = connection
    def __enter__(self): return self.connection
    def __exit__(self, typ, value, traceback): self.connection.commit() if typ is None else self.connection.rollback()


def _to_storage(settings: MediaToolsSettings) -> dict[str, object]: return asdict(settings)
def _from_dict(value: object) -> MediaToolsSettings:
    source = value if isinstance(value, dict) else {}
    def state(item: object) -> VerificationState:
        raw = item if isinstance(item, dict) else {}; return VerificationState(**{key: raw.get(key, getattr(VerificationState(), key)) for key in VerificationState.__dataclass_fields__})
    def part(cls, key: str):
        raw = source.get(key) if isinstance(source.get(key), dict) else {}; defaults = cls(); values = {field: raw.get(field, getattr(defaults, field)) for field in cls.__dataclass_fields__ if field != "verification"}; values["verification"] = state(raw.get("verification")); return cls(**values)
    return MediaToolsSettings(tts=part(TtsSettings, "tts"), stt=part(SttSettings, "stt"), ocr=part(OcrSettings, "ocr"))


def _verify_reset() -> VerificationState: return VerificationState()
def _tts_from_values(current: TtsSettings, values: dict[str, object]) -> TtsSettings:
    return replace(current, enabled=bool(values.get("enabled", current.enabled)), model_id=str(values.get("model_id", current.model_id)).strip(), device_map=str(values.get("device_map", current.device_map)).strip(), dtype=str(values.get("dtype", current.dtype)).strip().lower(), language=str(values.get("language", current.language)).strip(), speaker=str(values.get("speaker", current.speaker)).strip(), instruct=str(values.get("instruct", current.instruct)).strip(), attn_implementation=str(values.get("attn_implementation", current.attn_implementation)).strip(), local_files_only=bool(values.get("local_files_only", current.local_files_only)), verification=_verify_reset())
def _stt_from_values(current: SttSettings, values: dict[str, object]) -> SttSettings:
    return replace(current, enabled=bool(values.get("enabled", current.enabled)), executable_path=str(values.get("executable_path", current.executable_path)).strip(), model=str(values.get("model", current.model)).strip(), default_language=str(values.get("default_language", current.default_language)).strip(), verification=_verify_reset())
def _ocr_from_values(current: OcrSettings, values: dict[str, object]) -> OcrSettings:
    return replace(current, enabled=bool(values.get("enabled", current.enabled)), ollama_base_url=str(values.get("ollama_base_url", current.ollama_base_url)).strip().rstrip("/"), model_id=str(values.get("model_id", current.model_id)).strip(), timeout_seconds=float(values.get("timeout_seconds", current.timeout_seconds)), verification=_verify_reset())
def _validate(settings: MediaToolsSettings) -> MediaToolsSettings:
    if settings.tts.dtype not in {"", "float16", "float32", "bfloat16"}: raise ValueError("media_tools_tts_dtype_invalid")
    if settings.tts.enabled and not settings.tts.model_id: raise ValueError("media_tools_tts_model_required")
    if settings.stt.enabled and not settings.stt.model: raise ValueError("media_tools_stt_model_required")
    if settings.ocr.enabled and not settings.ocr.model_id: raise ValueError("media_tools_ocr_model_required")
    from urllib.parse import urlparse
    parsed = urlparse(settings.ocr.ollama_base_url)
    if settings.ocr.enabled and (parsed.scheme not in {"http", "https"} or not parsed.netloc): raise ValueError("media_tools_ocr_base_url_invalid")
    if not 1 <= settings.ocr.timeout_seconds <= 300: raise ValueError("media_tools_ocr_timeout_invalid")
    return settings
def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _platform_name() -> str:
    import platform
    return platform.system().lower()
def _say_available() -> bool:
    import shutil
    return _platform_name() == "darwin" and shutil.which("say") is not None
