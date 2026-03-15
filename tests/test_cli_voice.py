from __future__ import annotations

from pathlib import Path

from alphonse.agent import cli
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.voice_profiles import get_default_voice_profile
from alphonse.agent.nervous_system.voice_profiles import list_voice_profiles


def test_voice_enroll_sets_default_when_validation_passes(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "nerve-db"
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"wav")
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    monkeypatch.setattr(cli, "_validate_enrolled_voice_profile", lambda _name: {"output": {}, "exception": None})

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "voice",
            "enroll",
            "--name",
            "Alphonse",
            "--sample",
            str(sample),
            "--set-default",
        ]
    )
    cli._dispatch_command(args, db_path, parser)

    out = capsys.readouterr().out
    assert "Voice profile enrolled" in out
    rows = list_voice_profiles(limit=10)
    assert len(rows) == 1
    assert str(rows[0].get("status") or "") == "ready"
    default = get_default_voice_profile()
    assert isinstance(default, dict)
    assert str(default.get("name") or "") == "Alphonse"


def test_voice_delete_with_purge_removes_profile_row(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "nerve-db"
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"wav")
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    monkeypatch.setattr(cli, "_validate_enrolled_voice_profile", lambda _name: {"output": {}, "exception": None})

    parser = cli.build_parser()
    enroll = parser.parse_args(
        ["voice", "enroll", "--name", "Alphonse", "--sample", str(sample)]
    )
    cli._dispatch_command(enroll, db_path, parser)
    rows = list_voice_profiles(limit=10)
    assert len(rows) == 1

    delete = parser.parse_args(
        ["voice", "delete", "--profile", str(rows[0].get("profile_id")), "--purge-sample"]
    )
    cli._dispatch_command(delete, db_path, parser)

    out = capsys.readouterr().out
    assert "Deleted voice profile" in out
    assert list_voice_profiles(limit=10) == []
