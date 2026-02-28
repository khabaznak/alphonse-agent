from __future__ import annotations

from alphonse.integrations.homeassistant.smoke_toggle import run_toggle_smoke


class _FakeTool:
    def __init__(self, fn):
        self._fn = fn

    def execute(self, **kwargs):
        return self._fn(**kwargs)


class _FakeRegistry:
    def __init__(self) -> None:
        self._events = []

    def get(self, key: str):
        if key == "domotics.query":
            return _FakeTool(lambda **kwargs: {"status": "ok", "result": {"query": kwargs}, "error": None, "metadata": {}})
        if key == "domotics.subscribe":
            return _FakeTool(
                lambda **kwargs: {
                    "status": "ok",
                    "result": {"event_count": 2, "events": [{"entity_id": "input_boolean.alphonse_test_toggle"}]},
                    "error": None,
                    "metadata": {"subscribe_args": kwargs},
                }
            )
        if key == "domotics.execute":
            return _FakeTool(lambda **kwargs: {"status": "ok", "result": {"execute": kwargs}, "error": None, "metadata": {}})
        return None


def test_run_toggle_smoke_uses_domotics_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphonse.integrations.homeassistant.smoke_toggle.build_default_tool_registry",
        lambda: _FakeRegistry(),
    )

    result = run_toggle_smoke(entity_id="input_boolean.alphonse_test_toggle", duration_seconds=0.5)

    assert result["status"] == "ok"
    payload = result["result"] or {}
    assert payload.get("entity_id") == "input_boolean.alphonse_test_toggle"
    assert (payload.get("execute_on") or {}).get("status") == "ok"
    assert (payload.get("execute_off") or {}).get("status") == "ok"
