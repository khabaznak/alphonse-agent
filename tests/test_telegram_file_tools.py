from __future__ import annotations

from pathlib import Path

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


def test_telegram_get_file_meta_and_download(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(telegram_files, "TelegramAdapter", _FakeTelegramAdapter)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("ALPHONSE_TELEGRAM_FILES_DIR", str(tmp_path))

    meta_tool = telegram_files.TelegramGetFileMetaTool()
    meta = meta_tool.execute(file_id="f-1")
    assert meta["status"] == "ok"
    assert meta["file_path"] == "voice/file.ogg"

    dl_tool = telegram_files.TelegramDownloadFileTool()
    dl = dl_tool.execute(file_id="f-1")
    assert dl["status"] == "ok"
    assert Path(str(dl["local_path"])).exists()
    assert Path(str(dl["local_path"])).read_bytes() == b"fake-bytes"


def test_transcribe_and_analyze_fail_cleanly_without_openai(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(telegram_files, "TelegramAdapter", _FakeTelegramAdapter)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ALPHONSE_TELEGRAM_FILES_DIR", str(tmp_path))

    transcribe = telegram_files.TranscribeTelegramAudioTool().execute(file_id="f-audio")
    assert transcribe["status"] == "failed"
    assert transcribe["error"] == "openai_api_key_missing"

    analyze = telegram_files.AnalyzeTelegramImageTool().execute(file_id="f-image")
    assert analyze["status"] == "failed"
    assert analyze["error"] == "openai_api_key_missing"
