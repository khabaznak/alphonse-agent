from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import alphonse.agent.nervous_system.senses.telegram as telegram_sense
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal
from alphonse.agent.nervous_system.senses.telegram import TelegramSense
from alphonse.agent.nervous_system.telegram_invites import get_invite


class _FakeBus:
    def __init__(self) -> None:
        self.emitted: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self.emitted.append(signal)


class _FakeTelegramAdapter:
    def __init__(self, *, fail_download: bool = False) -> None:
        self.fail_download = fail_download
        self.file_ids: list[str] = []
        self.file_paths: list[str] = []

    def get_file(self, *, file_id: str) -> dict[str, Any]:
        self.file_ids.append(file_id)
        return {
            "file_path": f"voice/{file_id}.ogg",
            "file_size": 11,
            "file_unique_id": f"unique-{file_id}",
        }

    def download_file(self, *, file_path: str) -> bytes:
        self.file_paths.append(file_path)
        if self.fail_download:
            raise RuntimeError("download_failed")
        return b"FAKE-AUDIO"


class _FakeSttTool:
    def __init__(self, *, text: str = "transcribed voice note", fail: bool = False) -> None:
        self.text = text
        self.fail = fail

    def execute(self, *, asset_id: str, language_hint: str | None = None) -> dict[str, Any]:
        _ = language_hint
        if self.fail:
            return {
                "output": None,
                "exception": {"code": "transcription_failed", "retryable": True},
            }
        return {
            "output": {
                "asset_id": asset_id,
                "text": self.text,
                "segments": [{"start": 0, "end": 1, "text": self.text}],
            },
            "exception": None,
        }


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    sandbox_root = tmp_path / "sandbox"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SANDBOX_ROOT", str(sandbox_root))
    apply_schema(db_path)


def _patch_successful_ingestion(monkeypatch: pytest.MonkeyPatch, *, transcript: str = "transcribed voice note") -> list[dict[str, Any]]:
    registered: list[dict[str, Any]] = []
    monkeypatch.setattr(telegram_sense.identity, "resolve_user_id", lambda **_: "u-1")

    def _register_uploaded_asset(**kwargs: Any) -> dict[str, Any]:
        registered.append(kwargs)
        return {
            "asset_id": "asset-1",
            "kind": "audio",
            "mime": kwargs.get("mime_type") or "audio/ogg",
            "bytes": len(kwargs.get("content") or b""),
            "sha256": "sha",
        }

    monkeypatch.setattr(telegram_sense, "register_uploaded_asset", _register_uploaded_asset)
    monkeypatch.setattr(telegram_sense, "SttTranscribeTool", lambda: _FakeSttTool(text=transcript))
    return registered


def _emit_voice_message(sense: TelegramSense, *, text: str = "", kind: str = "voice") -> _FakeBus:
    bus = _FakeBus()
    sense._bus = bus  # type: ignore[attr-defined]
    sense._on_signal(  # type: ignore[attr-defined]
        Signal(
            type="external.telegram.message",
            payload={
                "text": text,
                "content_type": "media",
                "chat_type": "private",
                "chat_id": 8593816828,
                "from_user": 8593816828,
                "from_user_name": "Gabriela",
                "message_id": 2,
                "update_id": 31223564,
                "timestamp": 10.0,
                "attachments": [
                    {
                        "kind": kind,
                        "provider": "telegram",
                        "file_id": "voice-123",
                        "mime_type": "audio/ogg",
                        "duration_seconds": 4,
                    }
                ],
            },
            source="telegram",
        )
    )
    return bus


def test_voice_attachment_is_registered_transcribed_and_used_as_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    registered = _patch_successful_ingestion(monkeypatch, transcript="buy milk")
    adapter = _FakeTelegramAdapter()
    sense = TelegramSense()
    sense._adapter = adapter  # type: ignore[attr-defined]

    bus = _emit_voice_message(sense)

    assert len(bus.emitted) == 1
    payload = bus.emitted[0].payload
    assert payload["content"]["text"] == "buy milk"
    attachment = payload["content"]["attachments"][0]
    assert attachment["asset_id"] == "asset-1"
    assert attachment["asset_registration_status"] == "registered"
    assert attachment["transcription_status"] == "ok"
    assert attachment["transcript"] == "buy milk"
    assert adapter.file_ids == ["voice-123"]
    assert adapter.file_paths == ["voice/voice-123.ogg"]
    assert registered[0]["kind"] == "audio"
    assert registered[0]["owner_user_id"] == "u-1"
    assert registered[0]["channel_target"] == "8593816828"


def test_audio_attachment_follows_same_registration_and_transcription_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    _ = _patch_successful_ingestion(monkeypatch, transcript="audio transcript")
    sense = TelegramSense()
    sense._adapter = _FakeTelegramAdapter()  # type: ignore[attr-defined]

    bus = _emit_voice_message(sense, kind="audio")

    payload = bus.emitted[0].payload
    attachment = payload["content"]["attachments"][0]
    assert payload["content"]["text"] == "audio transcript"
    assert attachment["kind"] == "audio"
    assert attachment["asset_id"] == "asset-1"
    assert attachment["transcript"] == "audio transcript"


def test_text_plus_voice_keeps_text_primary_and_stores_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    _ = _patch_successful_ingestion(monkeypatch, transcript="voice detail")
    sense = TelegramSense()
    sense._adapter = _FakeTelegramAdapter()  # type: ignore[attr-defined]

    bus = _emit_voice_message(sense, text="typed prompt")

    payload = bus.emitted[0].payload
    assert payload["content"]["text"] == "typed prompt"
    attachment = payload["content"]["attachments"][0]
    assert attachment["transcript"] == "voice detail"
    ingestion = payload["metadata"]["normalized_metadata"]["telegram_attachment_ingestion"]
    assert ingestion["transcripts"][0]["text"] == "voice detail"


def test_download_failure_preserves_attachment_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    monkeypatch.setattr(telegram_sense.identity, "resolve_user_id", lambda **_: "u-1")
    monkeypatch.setattr(telegram_sense, "SttTranscribeTool", lambda: _FakeSttTool())
    sense = TelegramSense()
    sense._adapter = _FakeTelegramAdapter(fail_download=True)  # type: ignore[attr-defined]

    bus = _emit_voice_message(sense)

    payload = bus.emitted[0].payload
    assert payload["content"]["text"] == ""
    attachment = payload["content"]["attachments"][0]
    assert attachment["file_id"] == "voice-123"
    assert attachment["asset_registration_status"] == "failed"
    assert attachment["asset_registration_error"] == "download_failed"
    assert "asset_id" not in attachment


def test_transcription_failure_preserves_registered_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    _ = _patch_successful_ingestion(monkeypatch)
    monkeypatch.setattr(telegram_sense, "SttTranscribeTool", lambda: _FakeSttTool(fail=True))
    sense = TelegramSense()
    sense._adapter = _FakeTelegramAdapter()  # type: ignore[attr-defined]

    bus = _emit_voice_message(sense)

    payload = bus.emitted[0].payload
    assert payload["content"]["text"] == ""
    attachment = payload["content"]["attachments"][0]
    assert attachment["asset_id"] == "asset-1"
    assert attachment["asset_registration_status"] == "registered"
    assert attachment["transcription_status"] == "failed"
    assert attachment["transcription_error"] == "transcription_failed"


def test_invite_request_does_not_register_attachments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(telegram_sense, "register_uploaded_asset", lambda **kwargs: calls.append(kwargs))
    sense = TelegramSense()
    bus = _FakeBus()
    sense._bus = bus  # type: ignore[attr-defined]

    sense._on_signal(  # type: ignore[attr-defined]
        Signal(
            type="external.telegram.invite_request",
            payload={
                "chat_id": "-123",
                "from_user": "gaby",
                "from_user_name": "Gaby",
                "text": "hello",
                "attachments": [{"kind": "voice", "provider": "telegram", "file_id": "voice-1"}],
            },
            source="telegram",
        )
    )

    assert calls == []
    assert get_invite("-123")
