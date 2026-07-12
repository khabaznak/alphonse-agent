from __future__ import annotations

from typing import Any

import pytest

from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native import web
from alphonse.agent_v2.web_tools_settings import SQLiteWebToolsSettingsStore, WebToolsSettings


class _Response:
    status_code = 200
    url = "https://example.com/final"
    headers = {"content-type": "text/html"}
    text = "<html><title>Example</title><body><p>Hello world</p></body></html>"
    is_redirect = False
    is_permanent_redirect = False
    def json(self) -> Any: return {"results": [{"title": "Result", "url": "https://example.com", "content": "Snippet"}]}


def test_settings_defaults_validation_and_persistence() -> None:
    store = SQLiteWebToolsSettingsStore(":memory:")
    assert store.get().available is False
    with pytest.raises(ValueError, match="base_url_required"):
        store.save(WebToolsSettings(enabled=True))
    saved = store.save(WebToolsSettings(enabled=True, searxng_base_url="http://127.0.0.1:8080/"))
    assert saved.available is True
    assert saved.searxng_base_url == "http://127.0.0.1:8080"


def test_registry_hides_disabled_web_tools() -> None:
    disabled = build_native_tool_registry(WebToolsSettings())
    assert disabled.get("web_search") is None
    enabled = build_native_tool_registry(WebToolsSettings(enabled=True, searxng_base_url="http://127.0.0.1:8080"))
    assert enabled.get("web_search").tool_id == "native.web_search"
    assert enabled.get("web_fetch").tool_id == "native.web_fetch"


def test_search_normalizes_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web.requests, "get", lambda *args, **kwargs: _Response())
    result = web.execute_web_search({"query": "Alphonse", "limit": 1}, settings=WebToolsSettings(enabled=True, searxng_base_url="http://127.0.0.1:8080"))
    assert result["result_count"] == 1
    assert result["results"][0]["snippet"] == "Snippet"


def test_fetch_blocks_private_destinations() -> None:
    result = web.execute_web_fetch({"url": "http://127.0.0.1:8080"}, settings=WebToolsSettings(enabled=True, searxng_base_url="http://127.0.0.1:8080"))
    assert result["exception"]["code"] == "web_fetch_private_destination"


def test_fetch_extracts_public_html(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web.socket, "getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))])
    monkeypatch.setattr(web.requests, "get", lambda *args, **kwargs: _Response())
    result = web.execute_web_fetch({"url": "https://example.com", "max_chars": 10}, settings=WebToolsSettings(enabled=True, searxng_base_url="http://127.0.0.1:8080"))
    assert result["title"] == "Example"
    assert result["truncated"] is True


def test_fetch_revalidates_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    class Redirect(_Response):
        status_code = 302
        headers = {"location": "http://127.0.0.1/private"}
        is_redirect = True
        is_permanent_redirect = False
    original_getaddrinfo = web.socket.getaddrinfo
    monkeypatch.setattr(web.socket, "getaddrinfo", lambda host, *args, **kwargs: original_getaddrinfo(host, *args, **kwargs) if host == "127.0.0.1" else [(None, None, None, None, ("93.184.216.34", 443))])
    monkeypatch.setattr(web.requests, "get", lambda *args, **kwargs: Redirect())
    result = web.execute_web_fetch({"url": "https://example.com"}, settings=WebToolsSettings(enabled=True, searxng_base_url="http://127.0.0.1:8080"))
    assert result["exception"]["code"] == "web_fetch_private_destination"
