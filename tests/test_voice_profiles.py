from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.voice_profiles import create_voice_profile
from alphonse.agent.nervous_system.voice_profiles import delete_voice_profile
from alphonse.agent.nervous_system.voice_profiles import get_default_voice_profile
from alphonse.agent.nervous_system.voice_profiles import list_voice_profiles
from alphonse.agent.nervous_system.voice_profiles import purge_voice_profile_sample
from alphonse.agent.nervous_system.voice_profiles import resolve_voice_profile
from alphonse.agent.nervous_system.voice_profiles import set_default_voice_profile


def test_voice_profiles_persist_and_single_default(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    sample1 = tmp_path / "sample-1.wav"
    sample1.write_bytes(b"wav-1")
    sample2 = tmp_path / "sample-2.wav"
    sample2.write_bytes(b"wav-2")

    profile_a = create_voice_profile(
        {
            "profile_id": "p-a",
            "name": "Alphonse A",
            "source_sample_path": str(sample1),
            "backend": "qwen",
            "speaker_hint": "Ryan",
            "instruct": "calm",
            "is_default": True,
            "status": "ready",
        }
    )
    profile_b = create_voice_profile(
        {
            "profile_id": "p-b",
            "name": "Alphonse B",
            "source_sample_path": str(sample2),
            "backend": "qwen",
            "speaker_hint": "Jorge",
            "instruct": "clear",
            "is_default": False,
            "status": "ready",
        }
    )
    assert profile_a == "p-a"
    assert profile_b == "p-b"

    rows = list_voice_profiles(limit=10)
    assert len(rows) == 2
    assert sum(1 for row in rows if bool(row.get("is_default"))) == 1

    assert set_default_voice_profile("p-b") is True
    default = get_default_voice_profile()
    assert isinstance(default, dict)
    assert str(default.get("profile_id") or "") == "p-b"


def test_delete_and_resolve_voice_profile_and_purge_sample(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"wav")
    create_voice_profile(
        {
            "profile_id": "p-z",
            "name": "My Voice",
            "source_sample_path": str(sample),
            "backend": "qwen",
            "status": "ready",
        }
    )

    by_name = resolve_voice_profile("my voice")
    assert isinstance(by_name, dict)
    assert str(by_name.get("profile_id") or "") == "p-z"
    deleted = delete_voice_profile("p-z")
    assert isinstance(deleted, dict)
    assert resolve_voice_profile("p-z") is None
    assert purge_voice_profile_sample(str(deleted.get("source_sample_path") or "")) is True
