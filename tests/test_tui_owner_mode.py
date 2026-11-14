"""The terminal interface must not silently fall back to a different loop."""

import pytest

from alphonse.agent_v2.interfaces import tui


class Client:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def request(self, method):
        assert method == "experimental_owner_status"
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("ready", [True, False])
def test_owner_mode_is_refused_before_building_tui_runtime(monkeypatch, ready):
    monkeypatch.delenv("ALPHONSE_OWNER_ONLY_SIMPLE_LOOP", raising=False)
    monkeypatch.setattr(tui, "V2DaemonClient", lambda: Client({"processing_ready": ready}))
    monkeypatch.setattr(tui, "_daemon_is_available", lambda _: True)
    monkeypatch.setattr(tui.SQLiteIntegrationStore, "default", lambda: pytest.fail("opened runtime storage"))
    with pytest.raises(tui.TuiOwnerModeUnavailable, match="Use Desktop"):
        tui._build_textual_app_class()()


@pytest.mark.parametrize("error", ["owner_status_unavailable", "daemon_method_not_found: experimental_owner_status"])
def test_normal_and_older_daemon_remain_supported(monkeypatch, error):
    monkeypatch.delenv("ALPHONSE_OWNER_ONLY_SIMPLE_LOOP", raising=False)
    tui._check_tui_daemon_mode(Client(error=RuntimeError(error)), external_daemon=True)


def test_ambiguous_daemon_mode_fails_closed(monkeypatch):
    monkeypatch.delenv("ALPHONSE_OWNER_ONLY_SIMPLE_LOOP", raising=False)
    with pytest.raises(tui.TuiOwnerModeUnavailable, match="Unable to verify"):
        tui._check_tui_daemon_mode(Client(error=TimeoutError()), external_daemon=True)
    with pytest.raises(tui.TuiOwnerModeUnavailable, match="Unable to verify"):
        tui._check_tui_daemon_mode(Client(result={}), external_daemon=True)


def test_explicit_owner_mode_refuses_embedded_v3_fallback(monkeypatch):
    monkeypatch.setenv("ALPHONSE_OWNER_ONLY_SIMPLE_LOOP", "1")
    monkeypatch.setattr(tui, "V2DaemonClient", lambda: Client())
    monkeypatch.setattr(tui, "_daemon_is_available", lambda _: False)
    monkeypatch.setattr(tui.SQLiteIntegrationStore, "default", lambda: pytest.fail("opened runtime storage"))
    with pytest.raises(tui.TuiOwnerModeUnavailable, match="Use Desktop"):
        tui._build_textual_app_class()()


def test_main_reports_unsupported_mode_without_traceback(monkeypatch):
    class App:
        def __init__(self):
            raise tui.TuiOwnerModeUnavailable("Owner mode is unsupported; use Desktop.")

    monkeypatch.setattr(tui, "_build_textual_app_class", lambda: App)
    with pytest.raises(SystemExit, match="Owner mode is unsupported; use Desktop"):
        tui.main()
