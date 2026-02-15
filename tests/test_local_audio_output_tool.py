from __future__ import annotations

from types import SimpleNamespace

from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool
import alphonse.agent.tools.local_audio_output as lao


class _FakeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.pid = 4321
        self._returncode = returncode
        self.stderr = SimpleNamespace(read=lambda: "")

    def wait(self) -> int:
        return self._returncode


def test_local_audio_output_blocking_success_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")

    captured: dict[str, object] = {}

    def _fake_run(cmd, stdout, stderr, text, check):  # noqa: ANN001
        captured["cmd"] = cmd
        _ = (stdout, stderr, text, check)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(lao.subprocess, "run", _fake_run)

    tool = LocalAudioOutputSpeakTool()
    result = tool.execute(text="Hola mundo", voice="Monica", blocking=True)

    assert result["status"] == "ok"
    assert result["mode"] == "blocking"
    assert captured["cmd"] == ["say", "-v", "Monica", "Hola mundo"]


def test_local_audio_output_non_blocking_success_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")

    captured: dict[str, object] = {}

    def _fake_popen(cmd, stdout, stderr, text):  # noqa: ANN001
        captured["cmd"] = cmd
        _ = (stdout, stderr, text)
        return _FakeProc(returncode=0)

    class _ImmediateThread:
        def __init__(self, *, target, args, daemon):  # noqa: ANN001
            self._target = target
            self._args = args
            self.daemon = daemon

        def start(self) -> None:
            self._target(*self._args)

    monkeypatch.setattr(lao.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(lao.threading, "Thread", _ImmediateThread)

    tool = LocalAudioOutputSpeakTool()
    result = tool.execute(text="Hola", blocking=False)

    assert result["status"] == "ok"
    assert result["mode"] == "non_blocking"
    assert captured["cmd"] == ["say", "Hola"]


def test_local_audio_output_not_supported_non_macos(monkeypatch) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Linux")
    tool = LocalAudioOutputSpeakTool()
    result = tool.execute(text="Hola mundo")
    assert result["status"] == "failed"
    assert result["error"] == "local_audio_output_not_supported"


def test_local_audio_output_rejects_empty_text(monkeypatch) -> None:
    monkeypatch.setattr(lao.platform, "system", lambda: "Darwin")
    tool = LocalAudioOutputSpeakTool()
    result = tool.execute(text="   ")
    assert result["status"] == "failed"
    assert result["error"] == "text_required"
