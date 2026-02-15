from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.sandbox_dirs import get_sandbox_alias
from alphonse.agent.nervous_system.sandbox_dirs import resolve_sandbox_path


def test_default_sandbox_alias_exists_after_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SANDBOX_ROOT", str(tmp_path / "sandbox-root"))
    apply_schema(db_path)

    record = get_sandbox_alias("telegram_files")
    assert isinstance(record, dict)
    assert record.get("enabled") is True
    assert str(record.get("base_path") or "").endswith("sandbox-root/telegram_files")


def test_resolve_sandbox_path_rejects_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SANDBOX_ROOT", str(tmp_path / "sandbox-root"))
    apply_schema(db_path)

    with pytest.raises(ValueError, match="relative_path_invalid|sandbox_path_escape"):
        resolve_sandbox_path(alias="telegram_files", relative_path="../escape.txt")
