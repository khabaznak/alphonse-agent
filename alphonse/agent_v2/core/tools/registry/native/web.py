"""V2-native SearXNG search and safe public-web fetch tools."""

from __future__ import annotations

import ipaddress
import json
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.web_tools_settings import WebToolsSettings

WEB_SEARCH_TOOL_ID = "native.web_search"
WEB_FETCH_TOOL_ID = "native.web_fetch"
WEB_SEARCH_TOOL_NAME = "web_search"
WEB_FETCH_TOOL_NAME = "web_fetch"

WEB_SEARCH_ARGUMENT_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": False, "properties": {
    "query": {"type": "string", "description": "Search query."}, "limit": {"type": "integer", "description": "Maximum results (1-20)."},
    "categories": {"type": "string"}, "engines": {"type": "string"}, "language": {"type": "string"},
    "time_range": {"type": "string", "enum": ["day", "month", "year"]}, "safesearch": {"type": "integer", "enum": [0, 1, 2]},
}, "required": ["query"]}
WEB_FETCH_ARGUMENT_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": False, "properties": {
    "url": {"type": "string", "description": "Public http or https URL to read."}, "max_chars": {"type": "integer", "description": "Maximum returned text characters."},
}, "required": ["url"]}


def build_web_search_tool_definition(settings: WebToolsSettings) -> ToolDefinition:
    return ToolDefinition(ToolDescriptor(WEB_SEARCH_TOOL_ID, WEB_SEARCH_TOOL_NAME, ToolKind.NATIVE, "Search the web through the configured SearXNG service.", dict(WEB_SEARCH_ARGUMENT_SCHEMA), ("web", "search"), ("native", "web")), lambda arguments: execute_web_search(arguments, settings=settings), dict(WEB_SEARCH_ARGUMENT_SCHEMA), enabled=settings.available)


def build_web_fetch_tool_definition(settings: WebToolsSettings) -> ToolDefinition:
    return ToolDefinition(ToolDescriptor(WEB_FETCH_TOOL_ID, WEB_FETCH_TOOL_NAME, ToolKind.NATIVE, "Fetch readable text from a public web page after search.", dict(WEB_FETCH_ARGUMENT_SCHEMA), ("web", "fetch"), ("native", "web")), lambda arguments: execute_web_fetch(arguments, settings=settings), dict(WEB_FETCH_ARGUMENT_SCHEMA), enabled=settings.available)


def execute_web_search(arguments: dict[str, Any], *, settings: WebToolsSettings) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query: return _failed("web_search_invalid_query", "web_search requires a query.")
    if not settings.available: return _failed("web_tools_not_configured", "Web Tools are disabled or not configured.")
    params: dict[str, Any] = {"q": query, "format": "json"}
    for key in ("categories", "engines", "language", "time_range", "safesearch"):
        value = arguments.get(key)
        if value is not None and str(value).strip(): params[key] = str(value).strip()
    try:
        response = requests.get(urljoin(settings.searxng_base_url + "/", "search"), params=params, timeout=settings.search_timeout_seconds, headers={"Accept": "application/json"})
    except requests.Timeout: return _failed("searxng_timeout", "SearXNG request timed out.", retryable=True)
    except requests.RequestException as exc: return _failed("searxng_http_error", f"SearXNG request failed: {exc}", retryable=True)
    if response.status_code == 403: return _failed("searxng_json_disabled_or_forbidden", "SearXNG returned 403; JSON output may be disabled.", status_code=403)
    if response.status_code >= 400: return _failed("searxng_http_error", f"SearXNG returned status {response.status_code}.", retryable=response.status_code >= 500, status_code=response.status_code)
    try: payload = response.json()
    except (ValueError, json.JSONDecodeError): return _failed("searxng_invalid_json", "SearXNG did not return valid JSON.")
    if not isinstance(payload, dict): return _failed("searxng_invalid_json", "SearXNG JSON response was not an object.")
    limit = _limit(arguments.get("limit"), 20)
    results = [_result(item) for item in payload.get("results", []) if isinstance(item, dict)]
    results = [item for item in results if item["url"] or item["title"]][:limit]
    return {"query": query, "provider": "searxng", "result_count": len(results), "results": results, "answers": _values(payload.get("answers")), "suggestions": _values(payload.get("suggestions"))}


def execute_web_fetch(arguments: dict[str, Any], *, settings: WebToolsSettings) -> dict[str, Any]:
    url = str(arguments.get("url") or "").strip()
    if not settings.available: return _failed("web_tools_not_configured", "Web Tools are disabled or not configured.")
    try: _validate_public_url(url)
    except ValueError as exc: return _failed(str(exc), "web_fetch only permits public HTTP(S) destinations.")
    try:
        response = _safe_get(url, timeout=settings.fetch_timeout_seconds)
    except requests.Timeout: return _failed("web_fetch_timeout", "Web fetch timed out.", retryable=True)
    except requests.RequestException as exc: return _failed("web_fetch_http_error", f"Web fetch failed: {exc}", retryable=True)
    except ValueError as exc: return _failed(str(exc), "Web fetch redirect destination is not permitted.")
    if response.status_code >= 400: return _failed("web_fetch_http_error", f"Web fetch returned status {response.status_code}.", retryable=response.status_code >= 500, status_code=response.status_code)
    content_type = str(response.headers.get("content-type") or "")
    text = str(response.text or "")
    parser = _TextParser() if "html" in content_type.lower() or "<html" in text[:500].lower() else None
    if parser: parser.feed(text); title, text = parser.title, parser.text()
    else: title, text = "", _compact(text)
    max_chars = min(_limit(arguments.get("max_chars"), settings.fetch_max_chars), settings.fetch_max_chars)
    return {"url": url, "final_url": str(response.url or url), "status_code": response.status_code, "content_type": content_type, "title": title, "text": text[:max_chars].rstrip(), "truncated": len(text) > max_chars}


def _safe_get(url: str, *, timeout: float) -> requests.Response:
    current = url
    for _ in range(6):
        _validate_public_url(current)
        response = requests.get(current, timeout=timeout, headers={"User-Agent": "Alphonse/2.0 web_fetch", "Accept": "text/html,text/plain,*/*"}, allow_redirects=False)
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            if not location: return response
            current = urljoin(current, location); continue
        _validate_public_url(str(response.url or current))
        return response
    raise ValueError("web_fetch_too_many_redirects")


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}: raise ValueError("web_fetch_unsupported_scheme")
    if not parsed.hostname or parsed.username or parsed.password: raise ValueError("web_fetch_invalid_url")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"): raise ValueError("web_fetch_private_destination")
    try: addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except socket.gaierror as exc: raise ValueError("web_fetch_unresolvable_host") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global: raise ValueError("web_fetch_private_destination")


def _result(item: dict[str, Any]) -> dict[str, Any]:
    return {"title": str(item.get("title") or ""), "url": str(item.get("url") or ""), "snippet": str(item.get("content") or ""), "engines": _values(item.get("engines")), "category": str(item.get("category") or ""), "published_date": str(item.get("publishedDate") or "")}
def _values(value: Any) -> list[Any]: return list(value) if isinstance(value, list) else []
def _limit(value: Any, default: int) -> int:
    try: return max(1, min(int(value if value is not None else default), 20 if default == 20 else default))
    except (TypeError, ValueError): return default
def _failed(code: str, message: str, *, retryable: bool = False, status_code: int | None = None) -> dict[str, Any]:
    failure: dict[str, Any] = {"code": code, "message": message, "retryable": retryable}
    if status_code is not None: failure["status_code"] = status_code
    return {"output": None, "exception": failure}
def _compact(text: str) -> str: return " ".join(text.split())


class _TextParser(HTMLParser):
    def __init__(self) -> None: super().__init__(convert_charrefs=True); self.skip = 0; self.in_title = False; self.parts: list[str] = []; self.titles: list[str] = []
    @property
    def title(self) -> str: return _compact(" ".join(self.titles))
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        if tag in {"script", "style", "noscript"}: self.skip += 1
        elif tag == "title": self.in_title = True
        elif tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}: self.parts.append(" ")
    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip: self.skip -= 1
        elif tag == "title": self.in_title = False
    def handle_data(self, data: str) -> None:
        if self.skip: return
        if self.in_title: self.titles.append(data)
        self.parts.append(data)
    def text(self) -> str: return _compact(" ".join(self.parts))
