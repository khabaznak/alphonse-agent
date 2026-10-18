from __future__ import annotations

import base64
from pathlib import Path

import pytest

from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native.media import build_stt_transcribe_tool_definition, build_tts_render_tool_definition, verify_stt_recording
from alphonse.agent_v2.core.intelligence.task_state import TaskState


def _runtime(tmp_path: Path):
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    store = SQLiteMediaToolsSettingsStore(":memory:")
    return build_runtime_host(user_store=users, media_tools_settings_store=store), admin, store


def test_media_settings_save_invalidates_readiness_and_persists(tmp_path: Path) -> None:
    store = SQLiteMediaToolsSettingsStore(tmp_path / "media.sqlite3")
    saved = store.update("tts", {"enabled": True, "model_id": "local/qwen", "dtype": "float16", "speaker": "Alex"})
    assert saved.tts.enabled is True
    ready = store.mark_verification("tts", ready=True, preview="/tmp/sample.wav")
    assert ready.tts.available is True
    changed = store.update("tts", {"model_id": "local/new-qwen"})
    assert changed.tts.available is False
    assert SQLiteMediaToolsSettingsStore(tmp_path / "media.sqlite3").get().tts.model_id == "local/new-qwen"


def test_media_settings_unchanged_save_preserves_verification() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    store.update("stt", {"enabled": True})
    ready = store.mark_verification("stt", ready=True, preview="hola")

    saved = store.update("stt", {"enabled": True, "executable_path": ready.stt.executable_path, "model": "base", "default_language": ""})

    assert saved.stt.available is True
    assert saved.stt.verification.preview == "hola"


def test_media_settings_validate_backend_configuration() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    with pytest.raises(ValueError, match="media_tools_tts_dtype_invalid"):
        store.update("tts", {"enabled": True, "dtype": "int8"})
    with pytest.raises(ValueError, match="media_tools_kind_invalid"):
        store.update("ocr", {"enabled": True, "ollama_base_url": "not-a-url"})


def test_daemon_media_tools_require_admin_and_verification_updates_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, admin, _ = _runtime(tmp_path)
    daemon = V2Daemon(runtime)
    with pytest.raises(PermissionError, match="admin_required"):
        daemon.media_tools_settings(actor_user_id="not-admin")
    daemon.save_media_tools_settings(actor_user_id=admin.user_id, kind="tts", values={"enabled": True})
    monkeypatch.setattr("alphonse.agent_v2.daemon.verify_tts", lambda settings, sample_text: {"output": {"file_path": "/tmp/voice.wav"}, "exception": None})
    result = daemon.verify_media_tools(actor_user_id=admin.user_id, kind="tts")
    assert result["settings"]["tts"]["available"] is True
    assert result["result"]["exception"] is None


def test_recorded_stt_verification_marks_ready_and_keeps_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, admin, _ = _runtime(tmp_path)
    daemon = V2Daemon(runtime)
    daemon.save_media_tools_settings(actor_user_id=admin.user_id, kind="stt", values={"enabled": True})
    captured: dict[str, object] = {}

    def fake_verify(settings, *, audio_base64, mime_type, duration_ms):  # noqa: ANN001
        captured.update({"settings": settings, "audio_base64": audio_base64, "mime_type": mime_type, "duration_ms": duration_ms})
        return {"output": {"text": "hola Alphonse", "segments": []}, "exception": None}

    monkeypatch.setattr("alphonse.agent_v2.daemon.verify_stt_recording", fake_verify)
    result = daemon.verify_stt_recording(actor_user_id=admin.user_id, audio_base64=base64.b64encode(b"audio").decode(), mime_type="audio/webm", duration_ms=1_000)

    assert captured["mime_type"] == "audio/webm"
    assert result["settings"]["stt"]["available"] is True
    assert result["settings"]["stt"]["verification"]["preview"] == "hola Alphonse"


def test_recorded_stt_verification_failure_does_not_mark_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, admin, _ = _runtime(tmp_path)
    daemon = V2Daemon(runtime)
    daemon.save_media_tools_settings(actor_user_id=admin.user_id, kind="stt", values={"enabled": True})
    monkeypatch.setattr("alphonse.agent_v2.daemon.verify_stt_recording", lambda *args, **kwargs: {"output": None, "exception": {"code": "whisper_cli_not_found", "message": "Whisper CLI was not found."}})

    result = daemon.verify_stt_recording(actor_user_id=admin.user_id, audio_base64=base64.b64encode(b"audio").decode(), mime_type="audio/webm", duration_ms=1_000)

    assert result["settings"]["stt"]["available"] is False
    assert result["settings"]["stt"]["verification"]["error"] == "Whisper CLI was not found."


def test_recorded_stt_verification_validates_audio_and_removes_temporary_file(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SQLiteMediaToolsSettingsStore(":memory:").get().stt
    captured: dict[str, Path] = {}

    def fake_transcribe(_settings, *, asset_path: str):  # noqa: ANN001
        source = Path(asset_path)
        captured["source"] = source
        assert source.is_file()
        assert source.read_bytes() == b"recorded audio"
        return {"output": {"text": "hola", "segments": []}, "exception": None}

    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.transcribe_stt", fake_transcribe)
    result = verify_stt_recording(settings, audio_base64=base64.b64encode(b"recorded audio").decode(), mime_type="audio/webm;codecs=opus", duration_ms=1_500)

    assert result["output"]["text"] == "hola"
    assert not captured["source"].exists()
    assert verify_stt_recording(settings, audio_base64="", mime_type="audio/webm", duration_ms=1_000)["exception"]["code"] == "stt_recording_empty"
    assert verify_stt_recording(settings, audio_base64=base64.b64encode(b"x").decode(), mime_type="video/webm", duration_ms=1_000)["exception"]["code"] == "stt_recording_type_unsupported"
    assert verify_stt_recording(settings, audio_base64=base64.b64encode(b"x").decode(), mime_type="audio/webm", duration_ms=30_001)["exception"]["code"] == "stt_recording_duration_invalid"


def test_media_tools_are_not_capd_registered() -> None:
    tools = build_native_tool_registry()
    assert tools.get("tts_render") is None
    assert tools.get("stt_transcribe") is None
    assert tools.get("ocr_extract_text") is None
    assert tools.get("analyze_image") is None


def test_media_tool_contracts_exist_but_are_disabled_until_attachment_phase() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    settings = store.get()
    assert build_tts_render_tool_definition(settings.tts).enabled is False
    assert build_stt_transcribe_tool_definition(settings.stt).enabled is False


def test_attachment_manifest_is_task_visible_without_local_path() -> None:
    task = TaskState(
        goal="What is in this?",
        metadata={"attachments": [{"asset_id": "asset-1", "filename": "photo.jpg", "mime_type": "image/jpeg", "kind": "photo", "ingestion_status": "registered", "caption": "What is in this?", "path": "/private/secret/photo.jpg"}]},
    )

    prompt = task.to_markdown_prompt()

    assert "Asset ID: asset-1" in prompt
    assert "photo.jpg" in prompt
    assert "/private/secret/photo.jpg" not in prompt
