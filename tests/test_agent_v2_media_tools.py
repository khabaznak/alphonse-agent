from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native.media import build_ocr_extract_tool_definition, build_stt_transcribe_tool_definition, build_tts_render_tool_definition, build_analyze_image_tool_definition, analyze_image, analyze_task_image
from alphonse.agent_v2.assets import AttachmentDescriptor, SQLiteAssetStore
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


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
    assert tools.get("analyze_image") is None


def test_media_tool_contracts_exist_but_are_disabled_until_attachment_phase() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    settings = store.get()
    assert build_tts_render_tool_definition(settings.tts).enabled is False
    assert build_stt_transcribe_tool_definition(settings.stt).enabled is False
    assert build_ocr_extract_tool_definition(settings.ocr).enabled is False


def test_verified_vision_tool_analyzes_only_task_owned_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    store.update("ocr", {"enabled": True})
    settings = store.mark_verification("ocr", ready=True).ocr
    asset_store = SQLiteAssetStore(tmp_path / "assets.sqlite3", tmp_path / "assets")
    asset = asset_store.register_bytes(owner_user_id="alex", descriptor=AttachmentDescriptor("photo.jpg", "image/jpeg", 3), content=b"jpg", source="telegram")
    task = TaskState(user="alex", metadata={"asset_ids": [asset.asset_id]})
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue())
    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.analyze_image", lambda settings, asset_path, question, empty_code="image_analysis_empty": {"output": {"text": f"seen: {question}"}, "exception": None})

    result = analyze_task_image({"asset_id": asset.asset_id, "question": "What is shown?"}, context=context, settings=settings, asset_store=asset_store)

    assert result["output"]["text"] == "seen: What is shown?"
    with pytest.raises(ValueError, match="task_attachment_not_found"):
        analyze_task_image({"asset_id": "other", "question": "What is shown?"}, context=context, settings=settings, asset_store=asset_store)
    assert build_analyze_image_tool_definition(settings, asset_store).enabled is True


def test_attachment_manifest_is_task_visible_without_local_path() -> None:
    task = TaskState(
        goal="What is in this?",
        metadata={"attachments": [{"asset_id": "asset-1", "filename": "photo.jpg", "mime_type": "image/jpeg", "kind": "photo", "ingestion_status": "registered", "caption": "What is in this?", "path": "/private/secret/photo.jpg"}]},
    )

    prompt = task.to_markdown_prompt()

    assert "Asset ID: asset-1" in prompt
    assert "photo.jpg" in prompt
    assert "/private/secret/photo.jpg" not in prompt


def test_daemon_refreshes_vision_tool_after_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, admin, _ = _runtime(tmp_path)
    daemon = V2Daemon(runtime)
    daemon.save_media_tools_settings(actor_user_id=admin.user_id, kind="ocr", values={"enabled": True})
    assert runtime.core.tools.get("analyze_image") is None
    monkeypatch.setattr("alphonse.agent_v2.daemon.verify_ocr", lambda settings, sample_path: {"output": {"text": "ok"}, "exception": None})

    daemon.verify_media_tools(actor_user_id=admin.user_id, kind="ocr", sample="image.jpg")

    assert runtime.core.tools.get("analyze_image") is not None


def test_qwen_image_analysis_uses_caption_question_and_returns_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 200
        def json(self): return {"message": {"content": "A red bicycle."}}
    captured = {}
    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.requests.post", lambda url, json, timeout: captured.update({"url": url, "json": json, "timeout": timeout}) or Response())
    settings = SQLiteMediaToolsSettingsStore(":memory:").get().ocr
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")

    result = analyze_image(settings, asset_path=str(image), question="What is in this image?")

    assert result["output"]["text"] == "A red bicycle."
    assert captured["json"]["messages"][0]["content"] == "What is in this image?"
    assert captured["json"]["messages"][0]["images"]
