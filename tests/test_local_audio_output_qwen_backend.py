from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import alphonse.agent.tools.local_audio_output as lao
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool


def test_qwen_backend_render_reports_missing_dependencies(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_TTS_BACKEND", "qwen")
    monkeypatch.setattr(lao._QwenBackend, "_ensure_deps", lambda self: "deps missing")
    monkeypatch.setattr(
        lao._SayBackend,
        "render",
        lambda self, *, text, voice, output_dir, filename_prefix, format: lao._ok(
            {"file_path": str(tmp_path / "fallback.m4a"), "format": "m4a", "mime_type": "audio/mp4", "backend": "say"},
            tool="local_audio_output_render",
        ),
    )

    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex", output_dir=str(tmp_path), format="m4a")

    assert result["exception"] is None
    payload = result.get("output") or {}
    assert payload.get("backend") == "say"
    assert payload.get("fallback_from") == "qwen"
    assert payload.get("fallback_reason_code") == "qwen_backend_unavailable"


def test_qwen_backend_speak_uses_player_on_non_macos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_TTS_BACKEND", "qwen")
    monkeypatch.setattr(lao.platform, "system", lambda: "Linux")

    output_file = tmp_path / "sample.m4a"
    output_file.write_bytes(b"audio")

    def _fake_render(self, *, text, voice, output_dir, filename_prefix, format):  # noqa: ANN001
        _ = (text, voice, output_dir, filename_prefix, format)
        return lao._ok({"file_path": str(output_file), "format": "m4a", "mime_type": "audio/mp4"})

    monkeypatch.setattr(lao._QwenBackend, "render", _fake_render)
    monkeypatch.setattr(lao, "_resolve_audio_player", lambda _path: (["fake-player", str(output_file)], "fake-player"))

    calls: list[list[str]] = []

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        calls.append(list(cmd))
        _ = (stdout, stderr, text, check)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.subprocess, "run", _fake_run)

    tool = LocalAudioOutputSpeakTool()
    result = tool.execute(text="Hello", blocking=True)

    assert result["exception"] is None
    payload = result.get("output") or {}
    assert payload.get("backend") == "qwen"
    assert payload.get("player") == "fake-player"
    assert calls and calls[0][0] == "fake-player"
