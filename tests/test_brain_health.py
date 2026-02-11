from __future__ import annotations

from pathlib import Path
import sqlite3
import pytest

from alphonse.agent.cognition.brain_health import (
    BrainUnavailable,
    check_brain_health,
    require_brain_health,
)
from alphonse.agent.nervous_system.migrate import apply_schema


def test_check_brain_health_reports_unavailable_on_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db"
    with sqlite3.connect(db_path):
        pass
    health = check_brain_health(db_path)
    assert health.prompt_store_available is False


def test_require_brain_health_passes_after_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    health = require_brain_health(db_path)
    assert health.prompt_store_available is True


def test_require_brain_health_raises_when_unavailable(tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db"
    with sqlite3.connect(db_path):
        pass
    with pytest.raises(BrainUnavailable) as exc_info:
        require_brain_health(db_path)
    assert "unavailable" in str(exc_info.value)
