from __future__ import annotations

from types import SimpleNamespace

import alphonse.agent.tools.local_audio_output as lao
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool


def test_voice_selection_precedence_default_profile(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_QWEN_TTS_SPEAKER", "Ryan")
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
    assert selection.speaker == "Ryan"
    assert selection.instruct == "calm"


def test_voice_selection_precedence_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_QWEN_TTS_SPEAKER", "Ryan")
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
    assert selection.speaker == "Ryan"
    assert selection.profile_id is None


def test_voice_selection_named_profile_uses_env_speaker(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_QWEN_TTS_SPEAKER", "Ryan")
    monkeypatch.setattr(lao, "get_default_voice_profile", lambda: None)
    monkeypatch.setattr(
        lao,
        "resolve_voice_profile",
        lambda _ref: {
            "profile_id": "vp-samantha",
            "name": "Samantha",
            "speaker_hint": "Samantha",
            "instruct": "warm",
            "source_sample_path": "/tmp/samantha.wav",
        },
    )

    selection = lao._resolve_voice_selection("Samantha")
    assert selection.is_profile is True
    assert selection.profile_id == "vp-samantha"
    assert selection.profile_name == "Samantha"
    assert selection.speaker == "Ryan"
    assert selection.instruct == "warm"


def test_qwen_instruct_precedence_prefers_per_call(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_QWEN_TTS_INSTRUCT", "env calm")
    selection = lao._VoiceSelection(
        requested_voice="default",
        speaker="CustomSpeaker",
        instruct="profile calm",
        is_profile=True,
    )

    assert lao._resolve_qwen_instruct("per-call clear", selection=selection) == "per-call clear"


def test_qwen_instruct_precedence_falls_back_to_profile_then_env(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_QWEN_TTS_INSTRUCT", "env calm")
    selection_with_profile = lao._VoiceSelection(
        requested_voice="default",
        speaker="CustomSpeaker",
        instruct="profile calm",
        is_profile=True,
    )
    selection_without_profile = lao._VoiceSelection(
        requested_voice="default",
        speaker="Ryan",
        instruct=None,
        is_profile=False,
    )

    assert lao._resolve_qwen_instruct(None, selection=selection_with_profile) == "profile calm"
    assert lao._resolve_qwen_instruct(None, selection=selection_without_profile) == "env calm"


def test_qwen_failure_falls_back_to_say_for_speak(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_TTS_BACKEND", "qwen")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        lao._QwenBackend,
        "speak",
        lambda self, *, text, voice, blocking, volume, instruct=None: lao._failed("qwen_generate_failed", "boom"),
    )

    def _fake_say_speak(self, *, text, voice, blocking, volume, instruct=None):  # noqa: ANN001
        captured["voice"] = voice
        captured["instruct"] = instruct
        _ = (self, text, blocking, volume)
        return lao._ok({"backend": "say", "mode": "blocking"})

    monkeypatch.setattr(
        lao._SayBackend,
        "speak",
        _fake_say_speak,
    )

    tool = LocalAudioOutputSpeakTool()
    result = tool.execute(text="Hola", voice="Ryan", blocking=True)
    assert result["exception"] is None
    payload = result.get("output") or {}
    assert payload.get("backend") == "say"
    assert payload.get("fallback_from") == "qwen"
    assert payload.get("fallback_reason_code") == "qwen_generate_failed"
    assert captured["voice"] == "default"
    assert captured["instruct"] is None


def test_qwen_failure_falls_back_to_say_for_render(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_TTS_BACKEND", "qwen")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        lao._QwenBackend,
        "render",
        lambda self, *, text, voice, output_dir, filename_prefix, format, instruct=None: lao._failed("qwen_generate_failed", "boom"),
    )

    def _fake_say_render(self, *, text, voice, output_dir, filename_prefix, format, instruct=None):  # noqa: ANN001
        captured["voice"] = voice
        captured["instruct"] = instruct
        _ = (self, text, output_dir, filename_prefix, format)
        return lao._ok(
            {"backend": "say", "file_path": "/tmp/fallback.m4a", "format": "m4a", "mime_type": "audio/mp4"},
            tool="audio.render_local",
        )

    monkeypatch.setattr(
        lao._SayBackend,
        "render",
        _fake_say_render,
    )

    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola", voice="Ryan", format="m4a", instruct="per-call clear")
    assert result["exception"] is None
    payload = result.get("output") or {}
    assert payload.get("backend") == "say"
    assert payload.get("fallback_from") == "qwen"
    assert captured["voice"] == "default"
    assert captured["instruct"] == "per-call clear"


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
