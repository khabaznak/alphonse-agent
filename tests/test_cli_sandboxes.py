from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from alphonse.agent import cli


def test_cli_sandboxes_add_handles_mkdir_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    class _FakePath:
        def __init__(self, raw: str) -> None:
            self._raw = raw

        def expanduser(self):
            return self

        def resolve(self):
            return self

        def exists(self) -> bool:
            return False

        def is_dir(self) -> bool:
            return False

        def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
            _ = (parents, exist_ok)
            raise OSError("[Errno 30] Read-only file system")

        def __str__(self) -> str:
            return self._raw

    monkeypatch.setattr(cli, "Path", _FakePath)
    monkeypatch.setattr(cli, "get_sandbox_alias", lambda alias: None)
    monkeypatch.setattr(cli, "ensure_sandbox_alias", lambda **kwargs: None)

    cli._command_sandboxes(
        Namespace(
            sandboxes_command="add",
            alias="dumpster",
            path="/dumpster",
            description="test",
            yes=True,
        )
    )

    out = capsys.readouterr().out
    assert "Failed to create directory" in out
