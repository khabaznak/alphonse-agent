from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native.media import build_ocr_extract_tool_definition, build_stt_transcribe_tool_definition, build_tts_render_tool_definition


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


def test_media_settings_validate_backend_configuration() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    with pytest.raises(ValueError, match="media_tools_tts_dtype_invalid"):
        store.update("tts", {"enabled": True, "dtype": "int8"})
    with pytest.raises(ValueError, match="media_tools_ocr_base_url_invalid"):
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


def test_media_tools_are_not_capd_registered() -> None:
    tools = build_native_tool_registry()
    assert tools.get("tts_render") is None
    assert tools.get("stt_transcribe") is None
    assert tools.get("ocr_extract_text") is None


def test_media_tool_contracts_exist_but_are_disabled_until_attachment_phase() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    settings = store.get()
    assert build_tts_render_tool_definition(settings.tts).enabled is False
    assert build_stt_transcribe_tool_definition(settings.stt).enabled is False
    assert build_ocr_extract_tool_definition(settings.ocr).enabled is False
