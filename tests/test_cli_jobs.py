from __future__ import annotations

from argparse import Namespace

from alphonse.agent import cli


def test_cli_jobs_backfill_schedule_invokes_store(capsys, monkeypatch) -> None:
    called: dict[str, object] = {}

    class _FakeStore:
        def backfill_and_sync_jobs(self, *, user_id: str | None = None) -> dict[str, int]:
            called["user_id"] = user_id
            return {"scanned": 4, "updated": 2}

    monkeypatch.setattr(cli, "JobStore", _FakeStore)

    cli._command_jobs(Namespace(jobs_command="backfill-schedule", user_id="u1"))
    out = capsys.readouterr().out
    assert "scanned=4" in out
    assert "updated=2" in out
    assert called["user_id"] == "u1"
