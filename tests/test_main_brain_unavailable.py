from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

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


def test_main_starts_with_outbound_only_io_registry(monkeypatch, tmp_path: Path, caplog) -> None:
    db_path = tmp_path / "nerve-db"
    db_path.touch()

    monkeypatch.setattr(agent_main, "_resolve_nerve_db_path", lambda: db_path)
    monkeypatch.setattr(agent_main, "apply_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_main, "apply_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_main, "require_brain_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_main, "register_senses", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_main, "register_signals", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        agent_main,
        "all_senses",
        lambda: [
            SimpleNamespace(key="api"),
            SimpleNamespace(key="cli"),
            SimpleNamespace(key="telegram"),
        ],
    )
    monkeypatch.setattr(
        agent_main,
        "get_io_registry",
        lambda: SimpleNamespace(extremities={"cli": object(), "telegram": object()}),
    )

    class _FakeSenseManager:
        def __init__(self, *args, **kwargs) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

    class _FakeQueueRunner:
        enabled = True

        def __init__(self, *args, **kwargs) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

    class _FakeHeart:
        def run(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr(agent_main, "SenseManager", _FakeSenseManager)
    monkeypatch.setattr(agent_main, "PdcaQueueRunner", _FakeQueueRunner)
    monkeypatch.setattr(agent_main, "load_heart", lambda *_args, **_kwargs: _FakeHeart())

    caplog.set_level(logging.INFO)
    agent_main.main()

    assert any(
        "IO registry ready senses=api,cli,telegram extremities=cli,telegram" in rec.getMessage()
        for rec in caplog.records
    )
