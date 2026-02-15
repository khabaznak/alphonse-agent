from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from alphonse.agent.nervous_system.assets import register_uploaded_asset
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.tools.stt_transcribe import SttTranscribeTool
import alphonse.agent.tools.stt_transcribe as stt


def test_stt_transcribe_tool_transcribes_by_asset_id(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    sandbox_root = tmp_path / "sandbox"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SANDBOX_ROOT", str(sandbox_root))
    apply_schema(db_path)

    asset = register_uploaded_asset(
        content=b"FAKE-AUDIO",
        kind="audio",
        mime_type="audio/wav",
        owner_user_id="u1",
        provider="webui",
        channel_type="webui",
        channel_target="webui",
        original_filename="voice.wav",
    )
    asset_id = str(asset["asset_id"])

    monkeypatch.setattr(stt.shutil, "which", lambda _: "/usr/bin/whisper")

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        _ = (stdout, stderr, text, check)
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        out_file = out_dir / "voice.json"
        out_file.write_text(
            json.dumps(
                {
                    "text": "hola mundo",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hola mundo"}],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(stt.subprocess, "run", _fake_run)

    tool = SttTranscribeTool(model="tiny")
    result = tool.execute(asset_id=asset_id, language_hint="es-MX")

    assert result["status"] == "ok"
    assert result["asset_id"] == asset_id
    assert result["text"] == "hola mundo"
    assert isinstance(result.get("segments"), list)
    assert result["segments"][0]["text"] == "hola mundo"


def test_stt_transcribe_tool_fails_for_missing_asset() -> None:
    tool = SttTranscribeTool()
    result = tool.execute(asset_id="missing-asset-id")
    assert result["status"] == "failed"
    assert result["error"] == "asset_not_found"
