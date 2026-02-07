from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import CatalogUnavailable, IntentCatalogStore


def test_list_enabled_marks_unavailable_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path = tmp_path / "missing-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    store = IntentCatalogStore(str(db_path))
    with caplog.at_level("ERROR"):
        result = store.list_enabled()
    assert result == []
    assert store.available is False
    assert any("intent catalog unavailable" in rec.getMessage() for rec in caplog.records)


def test_list_enabled_raises_in_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "missing-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_ENV", "dev")
    store = IntentCatalogStore(str(db_path))
    with pytest.raises(CatalogUnavailable):
        store.list_enabled()


def test_get_marks_unavailable_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path = tmp_path / "missing-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    store = IntentCatalogStore(str(db_path))
    with caplog.at_level("ERROR"):
        result = store.get("timed_signals.create")
    assert result is None
    assert store.available is False
    assert any("intent catalog unavailable" in rec.getMessage() for rec in caplog.records)


def test_is_available_false_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "missing-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    store = IntentCatalogStore(str(db_path))
    assert store.is_available() is False
