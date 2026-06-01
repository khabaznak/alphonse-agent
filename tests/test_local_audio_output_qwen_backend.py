from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import alphonse.agent.tools.local_audio_output as lao
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool


def test_qwen_backend_render_reports_missing_dependencies(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_TTS_BACKEND", "qwen")
    monkeypatch.setattr(lao._QwenBackend, "_ensure_qwen_runtime", lambda self: "deps missing")
    monkeypatch.setattr(
        lao._SayBackend,
        "render",
        lambda self, *, text, voice, output_dir, filename_prefix, format, instruct=None: lao._ok(
            {"file_path": str(tmp_path / "fallback.m4a"), "format": "m4a", "mime_type": "audio/mp4", "backend": "say"},
            tool="audio.render_local",
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
    captured: dict[str, object] = {}

    def _fake_render(self, *, text, voice, output_dir, filename_prefix, format, instruct=None):  # noqa: ANN001
        captured["instruct"] = instruct
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
    result = tool.execute(text="Hello", blocking=True, instruct="Speak clearly.")

    assert result["exception"] is None
    payload = result.get("output") or {}
    assert payload.get("backend") == "qwen"
    assert payload.get("player") == "fake-player"
    assert calls and calls[0][0] == "fake-player"
    assert captured["instruct"] == "Speak clearly."


def test_qwen_backend_render_uses_per_call_instruct(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeSoundFile:
        @staticmethod
        def write(path: str, wav, sample_rate: int) -> None:  # noqa: ANN001
            captured["write"] = (path, wav, sample_rate)
            Path(path).write_bytes(b"wav")

    def _fake_generate_qwen_custom_voice(*, model, text, language, speaker, instruct, sample_path):  # noqa: ANN001
        captured["text"] = text
        captured["language"] = language
        captured["speaker"] = speaker
        captured["instruct"] = instruct
        captured["sample_path"] = sample_path
        return (["wav-frame"], 24000)

    monkeypatch.setattr(lao._QwenBackend, "_ensure_qwen_runtime", lambda self: None)
    monkeypatch.setattr(lao._QwenBackend, "_load_model", lambda self: object())
    monkeypatch.setattr(lao, "_generate_qwen_custom_voice", _fake_generate_qwen_custom_voice)
    monkeypatch.setattr(
        lao,
        "_convert_from_wav",
        lambda *, source_path, target_format: lao._ok(
            {"file_path": str(source_path), "format": target_format, "mime_type": "audio/wav"},
            tool="audio.render_local",
        ),
    )
    monkeypatch.setattr(lao, "_cleanup_rendered_audio", lambda *, root, keep_path: {"removed_by_age": 0, "removed_by_count": 0})

    backend = lao._QwenBackend()
    backend._soundfile = _FakeSoundFile
    result = backend.render(
        text="Hola",
        voice="Ryan",
        output_dir=str(tmp_path),
        filename_prefix="test",
        format="m4a",
        instruct="Speak slowly and clearly.",
    )

    assert result["exception"] is None
    assert captured["speaker"] == "Ryan"
    assert captured["instruct"] == "Speak slowly and clearly."


def test_qwen_model_load_uses_cached_snapshot_by_default(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    captured: dict[str, object] = {}

    def _fake_snapshot_download(model_id: str, *, local_files_only: bool) -> str:
        captured["snapshot_model_id"] = model_id
        captured["snapshot_local_only"] = local_files_only
        return str(snapshot)

    class _FakeHub:
        @staticmethod
        def snapshot_download(model_id: str, *, local_files_only: bool) -> str:
            return _fake_snapshot_download(model_id, local_files_only=local_files_only)

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs):  # noqa: ANN001
            captured["source"] = source
            captured["kwargs"] = dict(kwargs)
            return object()

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", _FakeHub)
    monkeypatch.setenv("ALPHONSE_QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    backend = lao._QwenBackend()
    backend._model_cls = _FakeModel

    assert backend._load_model() is not None

    assert captured["snapshot_model_id"] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert captured["snapshot_local_only"] is True
    assert captured["source"] == str(snapshot)
    assert (captured["kwargs"] or {}).get("local_files_only") is True


def test_qwen_model_load_can_allow_online_repo_resolution(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs):  # noqa: ANN001
            captured["source"] = source
            captured["kwargs"] = dict(kwargs)
            return object()

    monkeypatch.setenv("ALPHONSE_QWEN_TTS_MODEL", "Qwen/model")
    monkeypatch.setenv("ALPHONSE_QWEN_TTS_LOCAL_FILES_ONLY", "false")
    backend = lao._QwenBackend()
    backend._model_cls = _FakeModel

    assert backend._load_model() is not None

    assert captured["source"] == "Qwen/model"
    assert "local_files_only" not in (captured["kwargs"] or {})


def test_qwen_wav_to_m4a_prefers_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"wav")
    calls: list[list[str]] = []

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        calls.append(list(cmd))
        _ = (stdout, stderr, text, check)
        Path(cmd[-1]).write_bytes(b"m4a")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.shutil, "which", lambda name: "/usr/local/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(lao.subprocess, "run", _fake_run)

    result = lao._convert_from_wav(source_path=source, target_format="m4a")

    assert result["exception"] is None
    assert calls[0][:2] == ["/usr/local/bin/ffmpeg", "-y"]
    assert (result.get("output") or {}).get("format") == "m4a"
