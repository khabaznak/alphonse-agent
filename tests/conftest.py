from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _enable_json_ability_runtime_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    # Production default is disabled while tool inventory is being rebuilt.
    # Keep it enabled in tests unless a test overrides the env var explicitly.
    monkeypatch.setenv("ALPHONSE_ENABLE_JSON_ABILITY_RUNTIME", "true")
