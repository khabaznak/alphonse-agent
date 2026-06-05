from __future__ import annotations

from typing import Any

import pytest
import requests

from alphonse.agent.tools import searxng_search
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry import planner_tool_schemas
from alphonse.agent.tools.searxng_search import SearxngSearchTool
from alphonse.agent.tools.searxng_search import WebFetchTool


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        url: str = "https://example.test/final",
        headers: dict[str, str] | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.url = url
        self.headers = dict(headers or {})
        self._json_error = json_error

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def test_searxng_search_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)

    result = SearxngSearchTool().execute(query="test")

    assert result["output"] is None
    assert result["exception"]["code"] == "searxng_base_url_missing"


def test_searxng_search_builds_search_request_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("SEARXNG_BASE_URL", "https://searx.example")

    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(
            payload={
                "results": [
                    {
                        "title": "Result One",
                        "url": "https://example.com/one",
                        "content": "A snippet",
                        "engines": ["duckduckgo"],
                        "category": "general",
                        "score": 1.5,
                        "publishedDate": "2026-06-01",
                    },
                    {"title": "Result Two", "url": "https://example.com/two"},
                ],
                "answers": ["42"],
                "infoboxes": [{"title": "Info", "content": "Box"}],
                "corrections": ["corrected query"],
                "suggestions": ["suggested query"],
            },
            url=url,
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(searxng_search.requests, "get", _fake_get)

    result = SearxngSearchTool().execute(
        query="SearXNG API",
        limit=1,
        categories=["general", "it"],
        engines="duckduckgo",
        language="en",
        pageno=2,
        time_range="month",
        safesearch=1,
    )

    assert result["exception"] is None
    assert captured["url"] == "https://searx.example/search"
    params = captured["kwargs"]["params"]
    assert params["q"] == "SearXNG API"
    assert params["format"] == "json"
    assert params["categories"] == "general,it"
    assert params["engines"] == "duckduckgo"
    assert params["language"] == "en"
    assert params["pageno"] == "2"
    assert params["time_range"] == "month"
    assert params["safesearch"] == "1"
    output = result["output"]
    assert output["provider"] == "searxng"
    assert output["result_count"] == 1
    assert output["results"][0]["title"] == "Result One"
    assert output["results"][0]["snippet"] == "A snippet"
    assert output["results"][0]["published_date"] == "2026-06-01"
    assert output["answers"] == ["42"]
    assert output["infoboxes"] == [{"title": "Info", "content": "Box"}]
    assert output["corrections"] == ["corrected query"]
    assert output["suggestions"] == ["suggested query"]


def test_searxng_search_maps_403_to_json_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARXNG_BASE_URL", "https://searx.example")
    monkeypatch.setattr(
        searxng_search.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(status_code=403, text="forbidden"),
    )

    result = SearxngSearchTool().execute(query="test")

    assert result["output"] is None
    assert result["exception"]["code"] == "searxng_json_disabled_or_forbidden"
    assert result["exception"]["details"]["status_code"] == 403


def test_searxng_search_maps_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARXNG_BASE_URL", "https://searx.example")
    monkeypatch.setattr(
        searxng_search.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(json_error=ValueError("bad json")),
    )

    result = SearxngSearchTool().execute(query="test")

    assert result["output"] is None
    assert result["exception"]["code"] == "searxng_invalid_json"


def test_web_fetch_rejects_non_http_scheme() -> None:
    result = WebFetchTool().execute(url="file:///etc/passwd")

    assert result["output"] is None
    assert result["exception"]["code"] == "web_fetch_unsupported_scheme"


def test_web_fetch_extracts_title_text_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <html>
      <head><title>Example Page</title><style>.x{display:none}</style></head>
      <body>
        <h1>Heading</h1>
        <script>alert("ignored")</script>
        <p>First paragraph.</p>
        <p>Second paragraph has more words.</p>
      </body>
    </html>
    """
    monkeypatch.setattr(
        searxng_search.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            text=html,
            url="https://example.com/final",
            headers={"content-type": "text/html; charset=utf-8"},
        ),
    )

    result = WebFetchTool().execute(url="https://example.com/page", max_chars=120)

    assert result["exception"] is None
    output = result["output"]
    assert output["url"] == "https://example.com/page"
    assert output["final_url"] == "https://example.com/final"
    assert output["status_code"] == 200
    assert output["content_type"] == "text/html; charset=utf-8"
    assert output["title"] == "Example Page"
    assert "Heading First paragraph. Second paragraph" in output["text"]
    assert "ignored" not in output["text"]

    truncated = WebFetchTool().execute(url="https://example.com/page", max_chars=20)
    assert truncated["output"]["truncated"] is True
    assert len(truncated["output"]["text"]) <= 20


def test_web_fetch_maps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_timeout(*args: Any, **kwargs: Any) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(searxng_search.requests, "get", _raise_timeout)

    result = WebFetchTool().execute(url="https://example.com/page")

    assert result["output"] is None
    assert result["exception"]["code"] == "web_fetch_timeout"
    assert result["exception"]["retryable"] is True


def test_registry_exposes_web_tools_once() -> None:
    registry = build_default_tool_registry()
    schemas = [
        item["function"]["name"]
        for item in planner_tool_schemas(registry)
        if isinstance(item, dict)
    ]

    assert schemas.count("web.search") == 1
    assert schemas.count("web.fetch") == 1
