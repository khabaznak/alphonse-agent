"""Daemon-owned inference configuration and provider catalogs for v2."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRequest
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import OpenAICodexProvider
from alphonse.agent_v2.core.inference import OpenAICodexProviderConfig

OPENAI_CODEX_PROVIDER = "openai_codex"
CODEX_DEFAULT_MODEL = "__codex_default__"


@dataclass(frozen=True)
class InferenceSettingsRecord:
    provider_key: str = OPENAI_CODEX_PROVIDER
    model_id: str = ""
    validated_at: str = ""
    cli_version: str = ""
    validation_error: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "provider_key": self.provider_key,
            "model_id": self.model_id,
            "validated_at": self.validated_at,
            "cli_version": self.cli_version,
            "validation_error": self.validation_error,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    display_name: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id, "display_name": self.display_name, "description": self.description}


@dataclass(frozen=True)
class InferenceProviderDescriptor:
    provider_key: str
    display_name: str
    description: str

    def list_models(self) -> tuple[ModelOption, ...]:
        if self.provider_key != OPENAI_CODEX_PROVIDER:
            return ()
        return tuple(_codex_model_options())

    def build_router(self, model_id: str) -> InferenceRouter:
        if self.provider_key != OPENAI_CODEX_PROVIDER:
            raise ValueError(f"inference_provider_not_supported: {self.provider_key}")
        model = _normalize_model_id(model_id)
        return InferenceRouter(
            provider=OpenAICodexProvider(),
            default_profile=ModelProfile(
                provider=OPENAI_CODEX_PROVIDER,
                model=model,
                profile_id="chatgpt-plus-codex",
                supports_tool_calling=False,
                supports_structured_output=False,
                supports_json_mode=True,
                cost_tier="subscription",
            ),
        )

    def validate(self, model_id: str) -> tuple[str, str]:
        if self.provider_key != OPENAI_CODEX_PROVIDER:
            raise ValueError(f"inference_provider_not_supported: {self.provider_key}")
        model = _normalize_model_id(model_id)
        provider = OpenAICodexProvider(
            OpenAICodexProviderConfig(model=model or None, timeout_seconds=25.0, ephemeral=True)
        )
        profile = ModelProfile(provider=OPENAI_CODEX_PROVIDER, model=model, profile_id="validation")
        provider.generate_markdown(
            InferenceRequest(
                prompt="Reply with OK.",
                purpose=InferencePurpose.ACCEPTANCE_CRITERIA,
                model_profile=profile,
            )
        )
        return _codex_cli_version(), _now_iso()


class SQLiteInferenceSettingsStore:
    """Stores the daemon-wide inference selection, never provider credentials."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteInferenceSettingsStore":
        return cls(_default_db_path())

    def get(self) -> InferenceSettingsRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_inference_settings WHERE settings_id = 1").fetchone()
        if row is None:
            return InferenceSettingsRecord(model_id=os.getenv("OPENAI_CODEX_MODEL", "").strip())
        return InferenceSettingsRecord(
            provider_key=str(row["provider_key"]),
            model_id=str(row["model_id"]),
            validated_at=str(row["validated_at"]),
            cli_version=str(row["cli_version"]),
            validation_error=str(row["validation_error"]),
            updated_at=str(row["updated_at"]),
        )

    def save(self, record: InferenceSettingsRecord) -> InferenceSettingsRecord:
        provider = str(record.provider_key or "").strip().lower()
        if provider != OPENAI_CODEX_PROVIDER:
            raise ValueError(f"inference_provider_not_supported: {provider}")
        model = _normalize_model_id(record.model_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO v2_inference_settings
                  (settings_id, provider_key, model_id, validated_at, cli_version, validation_error, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                """,
                (provider, model, record.validated_at, record.cli_version, record.validation_error, _now_iso()),
            )
        return self.get()

    def _connect(self) -> sqlite3.Connection | "_ConnectionProxy":
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_inference_settings (
                  settings_id INTEGER PRIMARY KEY CHECK (settings_id = 1),
                  provider_key TEXT NOT NULL,
                  model_id TEXT NOT NULL DEFAULT '',
                  validated_at TEXT NOT NULL DEFAULT '',
                  cli_version TEXT NOT NULL DEFAULT '',
                  validation_error TEXT NOT NULL DEFAULT '',
                  updated_at TEXT NOT NULL
                ) STRICT;
                """
            )


def inference_provider_descriptors() -> tuple[InferenceProviderDescriptor, ...]:
    return (
        InferenceProviderDescriptor(
            provider_key=OPENAI_CODEX_PROVIDER,
            display_name="OpenAI Codex",
            description="Uses the signed-in local Codex CLI subscription provider.",
        ),
    )


def get_inference_provider(provider_key: str) -> InferenceProviderDescriptor:
    normalized = str(provider_key or "").strip().lower()
    for descriptor in inference_provider_descriptors():
        if descriptor.provider_key == normalized:
            return descriptor
    raise ValueError(f"inference_provider_not_supported: {normalized}")


def build_inference_router_from_settings(settings: InferenceSettingsRecord) -> InferenceRouter:
    return get_inference_provider(settings.provider_key).build_router(settings.model_id)


def validate_and_save_inference_settings(
    store: SQLiteInferenceSettingsStore,
    *,
    provider_key: str,
    model_id: str,
) -> InferenceSettingsRecord:
    descriptor = get_inference_provider(provider_key)
    try:
        cli_version, validated_at = descriptor.validate(model_id)
    except ValueError as exc:
        # Failed choices never become active, but retaining the diagnostic helps
        # the next configuration screen explain why validation did not succeed.
        current = store.get()
        store.save(
            InferenceSettingsRecord(
                provider_key=current.provider_key,
                model_id=current.model_id,
                validated_at=current.validated_at,
                cli_version=current.cli_version,
                validation_error=str(exc),
            )
        )
        raise
    return store.save(
        InferenceSettingsRecord(
            provider_key=descriptor.provider_key,
            model_id=_normalize_model_id(model_id),
            validated_at=validated_at,
            cli_version=cli_version,
        )
    )


def provider_status(provider_key: str) -> dict[str, Any]:
    descriptor = get_inference_provider(provider_key)
    cache_path = _codex_cache_path()
    metadata = _read_codex_cache_metadata(cache_path)
    return {
        "provider_key": descriptor.provider_key,
        "display_name": descriptor.display_name,
        "description": descriptor.description,
        "models": [option.to_dict() for option in descriptor.list_models()],
        "catalog_path": str(cache_path),
        "catalog_fetched_at": str(metadata.get("fetched_at") or ""),
        "catalog_cli_version": str(metadata.get("client_version") or ""),
        "cli_version": _codex_cli_version(),
    }


def _codex_model_options() -> list[ModelOption]:
    metadata = _read_codex_cache_metadata(_codex_cache_path())
    entries = metadata.get("models")
    if not isinstance(entries, list):
        return [ModelOption(CODEX_DEFAULT_MODEL, "Codex default", "Use the Codex CLI default model.")]
    options = [ModelOption(CODEX_DEFAULT_MODEL, "Codex default", "Use the Codex CLI default model.")]
    for entry in entries:
        if not isinstance(entry, dict) or str(entry.get("visibility") or "list") != "list":
            continue
        slug = str(entry.get("slug") or "").strip()
        if not slug:
            continue
        options.append(
            ModelOption(
                model_id=slug,
                display_name=str(entry.get("display_name") or slug).strip() or slug,
                description=str(entry.get("description") or "").strip(),
            )
        )
    return options


def _read_codex_cache_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _codex_cache_path() -> Path:
    return Path(os.getenv("ALPHONSE_V2_CODEX_MODELS_CACHE_PATH") or Path.home() / ".codex" / "models_cache.json")


def _codex_cli_version() -> str:
    cli_bin = os.getenv("OPENAI_CODEX_CLI_BIN", "codex")
    try:
        completed = subprocess.run([cli_bin, "--version"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return str(completed.stdout or "").strip() if completed.returncode == 0 else ""


def _normalize_model_id(value: str) -> str:
    normalized = str(value or "").strip()
    return "" if normalized == CODEX_DEFAULT_MODEL else normalized


def _default_db_path() -> Path:
    configured = os.getenv("ALPHONSE_V2_INFERENCE_DB_PATH") or os.getenv("ALPHONSE_V2_DB_PATH")
    return Path(configured) if configured else Path.home() / ".alphonse" / "v2-inference.sqlite3"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ConnectionProxy:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        return False
