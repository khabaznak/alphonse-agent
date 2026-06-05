from __future__ import annotations

import json
import os
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


class SearxngSearchTool:
    canonical_name: str = "web.search"
    capability: str = "web"

    def execute(
        self,
        *,
        query: str,
        limit: int | None = None,
        categories: str | list[str] | None = None,
        engines: str | list[str] | None = None,
        language: str | None = None,
        pageno: int | None = None,
        time_range: str | None = None,
        safesearch: int | str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = state
        rendered_query = str(query or "").strip()
        if not rendered_query:
            return _failed("web.search", "invalid_tool_arguments", "web.search requires `query`.")
        base_url = _read_base_url()
        if not base_url:
            return _failed(
                "web.search",
                "searxng_base_url_missing",
                "SEARXNG_BASE_URL is not configured. Set it in alphonse/agent/.env and restart Alphonse.",
            )

        params: dict[str, Any] = {
            "q": rendered_query,
            "format": "json",
        }
        _add_optional(params, "categories", _csv(categories))
        _add_optional(params, "engines", _csv(engines))
        _add_optional(params, "language", language)
        _add_optional(params, "pageno", _positive_int(pageno))
        _add_optional(params, "time_range", time_range)
        _add_optional(params, "safesearch", _safesearch_value(safesearch))

        try:
            response = requests.get(
                urljoin(base_url.rstrip("/") + "/", "search"),
                params=params,
                timeout=_read_positive_float("SEARXNG_TIMEOUT_SECONDS", 10.0),
                headers={"Accept": "application/json"},
            )
        except requests.Timeout:
            return _failed(
                "web.search",
                "searxng_http_error",
                "SearXNG request timed out.",
                retryable=True,
            )
        except requests.RequestException as exc:
            return _failed(
                "web.search",
                "searxng_http_error",
                f"SearXNG request failed: {exc}",
                retryable=True,
            )

        if response.status_code == 403:
            return _failed(
                "web.search",
                "searxng_json_disabled_or_forbidden",
                "SearXNG returned 403; JSON output may be disabled for this instance.",
                details={"status_code": response.status_code},
            )
        if response.status_code >= 400:
            return _failed(
                "web.search",
                "searxng_http_error",
                f"SearXNG request failed with status {response.status_code}.",
                retryable=500 <= response.status_code < 600,
                details={"status_code": response.status_code},
            )

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            return _failed(
                "web.search",
                "searxng_invalid_json",
                "SearXNG did not return valid JSON.",
                details={"status_code": response.status_code},
            )
        if not isinstance(payload, dict):
            return _failed(
                "web.search",
                "searxng_invalid_json",
                "SearXNG JSON response was not an object.",
            )

        max_results = _limit(limit)
        results = [_normalize_result(item) for item in _as_list(payload.get("results"))]
        results = [item for item in results if item.get("url") or item.get("title")]
        if max_results is not None:
            results = results[:max_results]

        return _ok(
            "web.search",
            {
                "query": rendered_query,
                "provider": "searxng",
                "result_count": len(results),
                "results": results,
                "answers": _lightweight_values(payload.get("answers")),
                "infoboxes": _lightweight_values(payload.get("infoboxes")),
                "corrections": _lightweight_values(payload.get("corrections")),
                "suggestions": _lightweight_values(payload.get("suggestions")),
            },
        )


class WebFetchTool:
    canonical_name: str = "web.fetch"
    capability: str = "web"

    def execute(
        self,
        *,
        url: str,
        max_chars: int | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = state
        rendered_url = str(url or "").strip()
        if not rendered_url:
            return _failed("web.fetch", "web_fetch_invalid_url", "web.fetch requires `url`.")
        parsed = urlparse(rendered_url)
        if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
            return _failed(
                "web.fetch",
                "web_fetch_unsupported_scheme",
                "web.fetch only supports http and https URLs.",
                details={"scheme": parsed.scheme},
            )
        if not parsed.scheme or not parsed.netloc:
            return _failed("web.fetch", "web_fetch_invalid_url", "web.fetch requires an absolute URL.")

        try:
            response = requests.get(
                rendered_url,
                timeout=_read_positive_float("ALPHONSE_WEB_FETCH_TIMEOUT_SECONDS", 10.0),
                headers={"User-Agent": "Alphonse/1.0 web.fetch", "Accept": "text/html,text/plain,*/*"},
            )
        except requests.Timeout:
            return _failed("web.fetch", "web_fetch_timeout", "web.fetch request timed out.", retryable=True)
        except requests.RequestException as exc:
            return _failed("web.fetch", "web_fetch_http_error", f"web.fetch request failed: {exc}", retryable=True)

        if response.status_code >= 400:
            return _failed(
                "web.fetch",
                "web_fetch_http_error",
                f"web.fetch request failed with status {response.status_code}.",
                retryable=500 <= response.status_code < 600,
                details={"status_code": response.status_code},
            )

        content_type = str(response.headers.get("content-type") or response.headers.get("Content-Type") or "")
        text = str(response.text or "")
        title = ""
        if "html" in content_type.lower() or "<html" in text[:500].lower():
            parser = _ReadableHtmlParser()
            parser.feed(text)
            title = parser.title.strip()
            readable = parser.text()
        else:
            readable = _compact_text(text)

        max_len = _max_chars(max_chars)
        truncated = len(readable) > max_len
        if truncated:
            readable = readable[:max_len].rstrip()

        return _ok(
            "web.fetch",
            {
                "url": rendered_url,
                "final_url": str(response.url or rendered_url),
                "status_code": int(response.status_code),
                "content_type": content_type,
                "title": title,
                "text": readable,
                "truncated": truncated,
            },
        )


class _ReadableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    @property
    def title(self) -> str:
        return _compact_text(" ".join(self._title_parts))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        name = tag.lower()
        if name in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif name == "title":
            self._in_title = True
        elif name in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self._text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif name == "title":
            self._in_title = False
        elif name in {"p", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self._text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = str(data or "").strip()
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        else:
            self._text_parts.append(text)

    def text(self) -> str:
        return _compact_text(" ".join(self._text_parts))


def _read_base_url() -> str:
    return str(os.getenv("SEARXNG_BASE_URL") or "").strip().rstrip("/")


def _add_optional(params: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    rendered = str(value).strip()
    if rendered:
        params[key] = rendered


def _csv(value: str | list[str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    items = [str(item or "").strip() for item in value]
    items = [item for item in items if item]
    return ",".join(items) if items else None


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safesearch_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed in {0, 1, 2} else None


def _limit(value: Any) -> int | None:
    parsed = _positive_int(value)
    if parsed is None:
        return 10
    return max(1, min(parsed, 50))


def _max_chars(value: Any) -> int:
    parsed = _positive_int(value)
    if parsed is None:
        parsed = int(_read_positive_float("ALPHONSE_WEB_FETCH_MAX_CHARS", 12000.0))
    return max(1, min(parsed, 50000))


def _read_positive_float(name: str, default: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return float(default)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default)
    return parsed if parsed > 0 else float(default)


def _normalize_result(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "title": _text_or_none(item.get("title")),
        "url": _text_or_none(item.get("url")),
        "snippet": _text_or_none(item.get("content") or item.get("snippet")),
        "content": _text_or_none(item.get("content")),
        "engines": _as_text_list(item.get("engines")),
        "category": _text_or_none(item.get("category")),
        "score": item.get("score") if isinstance(item.get("score"), (int, float)) else None,
        "published_date": _text_or_none(item.get("publishedDate") or item.get("published_date")),
    }


def _lightweight_values(value: Any) -> list[Any]:
    items = _as_list(value)
    out: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            out.append(
                {
                    key: item.get(key)
                    for key in (
                        "title",
                        "url",
                        "content",
                        "answer",
                        "suggestion",
                        "correction",
                        "infobox",
                        "engine",
                    )
                    if item.get(key) is not None
                }
            )
        elif item is not None:
            out.append(str(item))
    return out


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _as_text_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _text_or_none(value: Any) -> str | None:
    text = _compact_text(str(value)) if value is not None else ""
    return text or None


def _compact_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _ok(tool: str, output: dict[str, Any]) -> dict[str, Any]:
    return {"output": output, "exception": None, "metadata": {"tool": tool}}


def _failed(
    tool: str,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "output": None,
        "exception": {
            "code": code,
            "message": message,
            "retryable": bool(retryable),
            "details": dict(details or {}),
        },
        "metadata": {"tool": tool},
    }
