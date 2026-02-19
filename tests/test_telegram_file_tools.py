from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.tools import telegram_files


class _FakeTelegramAdapter:
    def __init__(self, config: dict) -> None:
        _ = config

    def get_file(self, file_id: str) -> dict:
        return {
            "file_id": file_id,
            "file_unique_id": "uq-1",
            "file_size": 12,
            "file_path": "voice/file.ogg",
        }

    def download_file(self, file_path: str) -> bytes:
        _ = file_path
        return b"fake-bytes"


class _FakeVisionResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.content = b"{}"

    def json(self) -> dict:
        return dict(self._body)


def test_telegram_get_file_meta_and_download(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(telegram_files, "TelegramAdapter", _FakeTelegramAdapter)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SANDBOX_ROOT", str(tmp_path / "sandbox-root"))
    apply_schema(db_path)

    meta_tool = telegram_files.TelegramGetFileMetaTool()
    meta = meta_tool.execute(file_id="f-1")
    assert meta["status"] == "ok"
    assert meta["file_path"] == "voice/file.ogg"

    dl_tool = telegram_files.TelegramDownloadFileTool()
    dl = dl_tool.execute(file_id="f-1")
    assert dl["status"] == "ok"
    assert dl["sandbox_alias"] == "telegram_files"
    rel = str(dl["relative_path"])
    saved = (tmp_path / "sandbox-root" / "telegram_files" / rel).resolve()
    assert saved.exists()
    assert saved.read_bytes() == b"fake-bytes"


def test_transcribe_and_analyze_with_ollama(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(telegram_files, "TelegramAdapter", _FakeTelegramAdapter)
    monkeypatch.setattr(telegram_files.shutil, "which", lambda _cmd: None)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setattr(
        telegram_files.requests,
        "post",
        lambda *args, **kwargs: _FakeVisionResponse(
            200,
            {"message": {"content": "Detected a handwritten note and one receipt."}},
        ),
    )
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SANDBOX_ROOT", str(tmp_path / "sandbox-root"))
    apply_schema(db_path)

    transcribe = telegram_files.TranscribeTelegramAudioTool().execute(file_id="f-audio")
    assert transcribe["status"] == "failed"
    assert transcribe["error"] == "whisper_cli_not_found"
    assert transcribe["sandbox_alias"] == "telegram_files"
    assert transcribe.get("relative_path")

    analyze = telegram_files.AnalyzeTelegramImageTool().execute(
        file_id=None,
        state={
            "channel_target": "8553589429",
            "provider_event": {
                "message": {
                    "photo": [
                        {"file_id": "small-photo"},
                        {"file_id": "best-photo"},
                    ]
                }
            },
        },
    )
    assert analyze["status"] == "ok"
    assert analyze["sandbox_alias"] == "telegram_files"
    assert str(analyze.get("relative_path") or "").startswith("users/8553589429/images/")
    assert analyze["analysis"] == "Detected a handwritten note and one receipt."


def test_vision_analyze_image_http_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        telegram_files.requests,
        "post",
        lambda *args, **kwargs: _FakeVisionResponse(500, {}),
    )
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SANDBOX_ROOT", str(tmp_path / "sandbox-root"))
    apply_schema(db_path)
    image_path = tmp_path / "sandbox-root" / "telegram_files" / "users" / "u1" / "images" / "sample.bin"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")

    result = telegram_files.VisionAnalyzeImageTool().execute(
        sandbox_alias="telegram_files",
        relative_path="users/u1/images/sample.bin",
    )
    assert result["status"] == "failed"
    assert result["error"] == "vision_http_500"
