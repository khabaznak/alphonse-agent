from __future__ import annotations

from types import SimpleNamespace

import alphonse.agent.tools.local_audio_output as lao
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool


def test_voice_selection_precedence_default_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        lao,
        "get_default_voice_profile",
        lambda: {
            "profile_id": "vp-1",
            "name": "Alphonse",
            "speaker_hint": "CustomSpeaker",
            "instruct": "calm",
            "source_sample_path": "/tmp/sample.wav",
        },
    )
    monkeypatch.setattr(lao, "resolve_voice_profile", lambda _ref: None)

    selection = lao._resolve_voice_selection("default")
    assert selection.is_profile is True
    assert selection.profile_id == "vp-1"
    assert selection.speaker == "CustomSpeaker"
    assert selection.instruct == "calm"


def test_voice_selection_precedence_explicit_override(monkeypatch) -> None:
    monkeypatch.setattr(
        lao,
        "get_default_voice_profile",
        lambda: {
            "profile_id": "vp-default",
            "name": "Alphonse",
            "speaker_hint": "DefaultSpeaker",
            "instruct": "calm",
            "source_sample_path": "/tmp/default.wav",
        },
    )
    monkeypatch.setattr(lao, "resolve_voice_profile", lambda _ref: None)

    selection = lao._resolve_voice_selection("WitchVoice")
    assert selection.is_profile is False
    assert selection.speaker == "WitchVoice"
    assert selection.profile_id is None


def test_qwen_failure_falls_back_to_say_for_speak(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_TTS_BACKEND", "qwen")
    monkeypatch.setattr(
        lao._QwenBackend,
        "speak",
        lambda self, *, text, voice, blocking, volume: lao._failed("qwen_generate_failed", "boom"),
    )
    monkeypatch.setattr(
        lao._SayBackend,
        "speak",
        lambda self, *, text, voice, blocking, volume: lao._ok({"backend": "say", "mode": "blocking"}),
    )

    tool = LocalAudioOutputSpeakTool()
    result = tool.execute(text="Hola", blocking=True)
    assert result["exception"] is None
    payload = result.get("output") or {}
    assert payload.get("backend") == "say"
    assert payload.get("fallback_from") == "qwen"
    assert payload.get("fallback_reason_code") == "qwen_generate_failed"


def test_qwen_failure_falls_back_to_say_for_render(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_TTS_BACKEND", "qwen")
    monkeypatch.setattr(
        lao._QwenBackend,
        "render",
        lambda self, *, text, voice, output_dir, filename_prefix, format: lao._failed("qwen_generate_failed", "boom"),
    )
    monkeypatch.setattr(
        lao._SayBackend,
        "render",
        lambda self, *, text, voice, output_dir, filename_prefix, format: lao._ok(
            {"backend": "say", "file_path": "/tmp/fallback.m4a", "format": "m4a", "mime_type": "audio/mp4"},
            tool="audio.render_local",
        ),
    )

    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola", format="m4a")
    assert result["exception"] is None
    payload = result.get("output") or {}
    assert payload.get("backend") == "say"
    assert payload.get("fallback_from") == "qwen"


def test_qwen_generate_custom_voice_uses_reference_sample_when_supported(tmp_path, monkeypatch) -> None:
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"wav")

    captured: dict[str, object] = {}

    class _Model:
        def generate_custom_voice(
            self,
            *,
            text,
            language,
            speaker,
            instruct,
            reference_audio_path=None,
        ):  # noqa: ANN001
            captured["reference_audio_path"] = reference_audio_path
            _ = (text, language, speaker, instruct)
            return (["wav"], 24000)

    wavs, sr = lao._generate_qwen_custom_voice(
        model=_Model(),
        text="hello",
        language="Auto",
        speaker="Ryan",
        instruct=None,
        sample_path=str(sample),
    )
    assert wavs == ["wav"]
    assert sr == 24000
    assert str(captured.get("reference_audio_path") or "") == str(sample.resolve())


def test_qwen_generate_custom_voice_ignores_reference_when_unsupported(tmp_path, monkeypatch) -> None:
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"wav")

    called = SimpleNamespace(ok=False)

    class _Model:
        def generate_custom_voice(self, *, text, language, speaker, instruct):  # noqa: ANN001
            _ = (text, language, speaker, instruct)
            called.ok = True
            return (["wav"], 24000)

    wavs, _ = lao._generate_qwen_custom_voice(
        model=_Model(),
        text="hello",
        language="Auto",
        speaker="Ryan",
        instruct=None,
        sample_path=str(sample),
    )
    assert called.ok is True
    assert wavs == ["wav"]
