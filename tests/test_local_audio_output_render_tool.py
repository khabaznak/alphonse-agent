from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import alphonse.agent.tools.local_audio_output as lao
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool


def test_local_audio_output_render_m4a_on_macos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")
    calls: list[list[str]] = []

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        calls.append(list(cmd))
        _ = (stdout, stderr, text, check)
        if cmd and cmd[0] == "say":
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(b"fake-aiff")
        elif cmd and cmd[0] == "afconvert":
            Path(cmd[-1]).write_bytes(b"fake-m4a")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.subprocess, "run", _fake_run)

    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex", output_dir=str(tmp_path), format="m4a")

    assert result["status"] == "ok"
    payload = result["result"]
    assert str(payload["file_path"]).endswith(".m4a")
    assert payload["mime_type"] == "audio/mp4"
    assert calls[0][0] == "say"
    assert calls[1][0] == "afconvert"


def test_local_audio_output_render_rejects_non_macos(monkeypatch) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Linux")
    tool = LocalAudioOutputRenderTool()
    result = tool.execute(text="Hola Alex")
    assert result["status"] == "failed"
    assert (result.get("error") or {}).get("code") == "local_audio_output_not_supported"
