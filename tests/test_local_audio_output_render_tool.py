from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import alphonse.agent.tools.local_audio_output as lao
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool


def test_local_audio_output_render_m4a_on_macos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")
    calls: list[list[str]] = []

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        calls.append(list(cmd))
        _ = (stdout, stderr, text, check)
        if cmd and cmd[0] == "say":
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(b"fake-aiff")
        elif cmd and cmd[0] == "afconvert":
            Path(cmd[-1]).write_bytes(b"fake-m4a")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.subprocess, "run", _fake_run)

    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex", output_dir=str(tmp_path), format="m4a")

    assert result["status"] == "ok"
    payload = result["result"]
    assert str(payload["file_path"]).endswith(".m4a")
    assert payload["mime_type"] == "audio/mp4"
    assert calls[0][0] == "say"
    assert calls[1][0] == "afconvert"


def test_local_audio_output_render_ogg_on_macos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lao.shutil, "which", lambda name: "/usr/local/bin/ffmpeg" if name == "ffmpeg" else None)
    calls: list[list[str]] = []

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        calls.append(list(cmd))
        _ = (stdout, stderr, text, check)
        if cmd and cmd[0] == "say":
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(b"fake-aiff")
        elif cmd and "ffmpeg" in str(cmd[0]):
            Path(cmd[-1]).write_bytes(b"fake-ogg")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.subprocess, "run", _fake_run)

    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex", output_dir=str(tmp_path), format="ogg")

    assert result["status"] == "ok"
    payload = result["result"]
    assert str(payload["file_path"]).endswith(".ogg")
    assert payload["mime_type"] == "audio/ogg"
    assert calls[0][0] == "say"
    assert "ffmpeg" in str(calls[1][0])


def test_local_audio_output_render_ogg_requires_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lao.shutil, "which", lambda _name: None)

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        _ = (stdout, stderr, text, check)
        if cmd and cmd[0] == "say":
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(b"fake-aiff")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.subprocess, "run", _fake_run)
    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex", output_dir=str(tmp_path), format="ogg")
    assert result["status"] == "failed"
    assert str((result.get("error") or {}).get("code") or "") == "ffmpeg_not_installed"


def test_local_audio_output_render_rejects_non_macos(monkeypatch) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Linux")
    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex")
    assert result["status"] == "failed"
    assert (result.get("error") or {}).get("code") == "local_audio_output_not_supported"


def test_local_audio_output_render_defaults_to_authorized_workdir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lao, "PRIMARY_WORKDIR_ALIASES", ("main",))
    monkeypatch.setattr(
        lao,
        "get_sandbox_alias",
        lambda alias: {"alias": alias, "enabled": True, "base_path": str(tmp_path / "main-workdir")},
    )

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        _ = (stdout, stderr, text, check)
        if cmd and cmd[0] == "say":
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(b"fake-aiff")
        elif cmd and cmd[0] == "afconvert":
            Path(cmd[-1]).write_bytes(b"fake-m4a")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.subprocess, "run", _fake_run)

    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex")

    assert result["status"] == "ok"
    rendered_path = Path(str((result.get("result") or {}).get("file_path") or ""))
    assert rendered_path.parent == (tmp_path / "main-workdir" / "audio_output").resolve()


def test_local_audio_output_render_prunes_old_and_excess_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("ALPHONSE_AUDIO_MAX_FILES", "2")
    monkeypatch.setenv("ALPHONSE_AUDIO_MAX_AGE_HOURS", "1")

    old_a = tmp_path / "response-old-a.m4a"
    old_b = tmp_path / "response-old-b.m4a"
    recent = tmp_path / "response-recent.m4a"
    old_a.write_bytes(b"old-a")
    old_b.write_bytes(b"old-b")
    recent.write_bytes(b"recent")

    old_ts = 100.0
    recent_ts = 5000.0
    old_a.touch()
    old_b.touch()
    recent.touch()
    old_a_ts = (old_ts, old_ts)
    old_b_ts = (old_ts + 10.0, old_ts + 10.0)
    recent_pair = (recent_ts, recent_ts)
    import os

    os.utime(old_a, old_a_ts)
    os.utime(old_b, old_b_ts)
    os.utime(recent, recent_pair)

    fake_now = datetime(1970, 1, 1, 2, 0, 0, tzinfo=timezone.utc)

    class _FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            return fake_now

    monkeypatch.setattr(lao, "datetime", _FakeDateTime)

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        _ = (stdout, stderr, text, check)
        if cmd and cmd[0] == "say":
            out_idx = cmd.index("-o") + 1
            path = Path(cmd[out_idx])
            path.write_bytes(b"new-aiff")
        elif cmd and cmd[0] == "afconvert":
            Path(cmd[-1]).write_bytes(b"new-m4a")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.subprocess, "run", _fake_run)

    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex", output_dir=str(tmp_path), format="m4a")

    assert result["status"] == "ok"
    retention = (result.get("result") or {}).get("retention") or {}
    assert int(retention.get("removed_by_age") or 0) >= 2
    remaining = sorted(tmp_path.glob("*.m4a"))
    assert len(remaining) <= 2
