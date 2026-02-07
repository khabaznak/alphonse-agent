from __future__ import annotations

import logging
from pathlib import Path

from alphonse.agent.cognition.brain_health import BrainUnavailable
from alphonse.agent import main as agent_main


def test_main_exits_gracefully_when_brain_unavailable(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    db_path = tmp_path / "nerve-db"
    db_path.touch()

    monkeypatch.setattr(agent_main, "_resolve_nerve_db_path", lambda: db_path)
    monkeypatch.setattr(agent_main, "apply_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_main, "apply_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_main, "init_settings_db", lambda: None)

    def _raise_unavailable(*_args, **_kwargs):
        raise BrainUnavailable("intent catalog unavailable")

    monkeypatch.setattr(agent_main, "require_brain_health", _raise_unavailable)

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("runtime components should not start when brain is unavailable")

    monkeypatch.setattr(agent_main, "register_senses", _must_not_run)
    monkeypatch.setattr(agent_main, "register_signals", _must_not_run)

    caplog.set_level(logging.CRITICAL)
    agent_main.main()

    assert any(
        "Brain health check failed. Shutting down gracefully" in rec.getMessage()
        for rec in caplog.records
    )
