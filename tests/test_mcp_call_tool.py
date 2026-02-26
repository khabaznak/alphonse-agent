from __future__ import annotations

import json
from pathlib import Path

import pytest

import alphonse.agent.tools.mcp_call_tool as mcp_call_tool_module
from alphonse.agent.tools.mcp_call_tool import McpCallTool
from alphonse.agent.tools.mcp_connector import McpConnector


class _DummyTerminal:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute_with_policy(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "status": "ok",
            "result": {
                "exit_code": 0,
                "stdout": "ok",
                "stderr": "",
            },
            "error": None,
            "metadata": {"policy_decision": "allow"},
        }


def test_mcp_call_executes_with_policy_envelope(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "ops")
    monkeypatch.setattr(mcp_call_tool_module, "_allowed_roots", lambda: [str(tmp_path)])
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp", "chrome-mcp"],
                "aliases": ["chrome-devtools"],
                "operations": {
                    "open_page": {
                        "key": "open_page",
                        "description": "open page",
                        "command_template": "open {url}",
                        "required_args": ["url"],
                    }
                },
                "npx_package_fallback": "chrome-devtools-mcp",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    dummy_terminal = _DummyTerminal()
    tool = McpCallTool(connector=McpConnector(terminal=dummy_terminal))

    result = tool.execute(
        profile="chrome",
        operation="open_page",
        arguments={"url": "https://example.com"},
        cwd=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert dummy_terminal.calls
    call = dummy_terminal.calls[0]
    command = str(call.get("command") or "")
    assert "chrome-devtools-mcp" in command or "chrome-mcp" in command or "npx -y chrome-devtools-mcp" in command
    assert "fi search" not in command
    assert 'eval "$MCP_BIN ' in command
    metadata = result.get("metadata")
    assert isinstance(metadata, dict)
    envelope = metadata.get("policy_envelope")
    assert isinstance(envelope, dict)
    assert envelope.get("execution_surface") == "mcp"
    assert envelope.get("profile") == "chrome"
    assert envelope.get("operation") == "open_page"


def test_mcp_call_rejects_unknown_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "ops")
    monkeypatch.setattr(mcp_call_tool_module, "_allowed_roots", lambda: [str(tmp_path)])
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(tmp_path / "profiles-empty"))
    tool = McpCallTool(connector=McpConnector(terminal=_DummyTerminal()))

    result = tool.execute(
        profile="unknown",
        operation="web_search",
        arguments={"query": "Veloswim"},
        cwd=str(tmp_path),
    )

    assert result["status"] == "failed"
    error = result.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "mcp_profile_not_found"


def test_mcp_call_normalizes_legacy_query_and_ignores_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "ops")
    monkeypatch.setattr(mcp_call_tool_module, "_allowed_roots", lambda: [str(tmp_path)])
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp"],
                "operations": {
                    "open_page": {
                        "key": "open_page",
                        "description": "open page",
                        "command_template": "open {url}",
                        "required_args": ["url"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    dummy_terminal = _DummyTerminal()
    tool = McpCallTool(connector=McpConnector(terminal=dummy_terminal))

    result = tool.execute(
        profile="chrome",
        operation="open_page",
        url="https://example.com",
        mode="headless",
        cwd=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert dummy_terminal.calls
    command = str(dummy_terminal.calls[0].get("command") or "")
    assert "https://example.com" in command


def test_mcp_call_normalizes_nested_args_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "ops")
    monkeypatch.setattr(mcp_call_tool_module, "_allowed_roots", lambda: [str(tmp_path)])
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp"],
                "operations": {
                    "open_page": {
                        "key": "open_page",
                        "description": "open page",
                        "command_template": "open {url}",
                        "required_args": ["url"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    dummy_terminal = _DummyTerminal()
    tool = McpCallTool(connector=McpConnector(terminal=dummy_terminal))

    result = tool.execute(
        args={
            "profile": "chrome",
            "operation": "open_page",
            "url": "https://example.org",
        },
        cwd=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert dummy_terminal.calls
    command = str(dummy_terminal.calls[0].get("command") or "")
    assert "https://example.org" in command


def test_mcp_call_uses_short_default_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "ops")
    monkeypatch.setattr(mcp_call_tool_module, "_allowed_roots", lambda: [str(tmp_path)])
    monkeypatch.delenv("ALPHONSE_MCP_DEFAULT_TIMEOUT_SECONDS", raising=False)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp"],
                "operations": {
                    "open_page": {
                        "key": "open_page",
                        "description": "open url",
                        "command_template": "open {url}",
                        "required_args": ["url"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    dummy_terminal = _DummyTerminal()
    tool = McpCallTool(connector=McpConnector(terminal=dummy_terminal))

    result = tool.execute(
        profile="chrome",
        operation="open_page",
        arguments={"url": "https://example.com"},
        cwd=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert dummy_terminal.calls
    timeout_value = float(dummy_terminal.calls[0].get("timeout_seconds") or 0)
    assert timeout_value == 45.0


def test_mcp_call_fast_fails_contract_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "ops")
    monkeypatch.setattr(mcp_call_tool_module, "_allowed_roots", lambda: [str(tmp_path)])
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp"],
                "operations": {
                    "web_search": {
                        "key": "web_search",
                        "description": "search web",
                        "command_template": "search {query}",
                        "required_args": ["query"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    dummy_terminal = _DummyTerminal()
    tool = McpCallTool(connector=McpConnector(terminal=dummy_terminal))

    result = tool.execute(
        profile="chrome",
        operation="web_search",
        arguments={"query": "Veloswim"},
        cwd=str(tmp_path),
    )

    assert result["status"] == "failed"
    assert not dummy_terminal.calls
    error = result.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "mcp_operation_contract_mismatch"


def test_mcp_call_routes_unknown_operation_to_native_connector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "ops")
    monkeypatch.setattr(mcp_call_tool_module, "_allowed_roots", lambda: [str(tmp_path)])
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp"],
                "operations": {},
                "metadata": {"native_tools": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))

    connector = McpConnector(terminal=_DummyTerminal())
    called: dict[str, object] = {}

    def _fake_native(**kwargs):
        called.update(kwargs)
        return {
            "status": "ok",
            "result": {"tools": ["navigate"]},
            "error": None,
            "metadata": {"tool": "mcp_call", "mcp_native": True},
        }

    monkeypatch.setattr(connector, "_execute_native", _fake_native)
    tool = McpCallTool(connector=connector)

    result = tool.execute(
        profile="chrome",
        operation="list_tools",
        arguments={},
        cwd=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert called.get("operation_key") == "list_tools"


class _FakeNativeClient:
    starts = 0
    closes = 0
    initializes = 0
    calls: list[tuple[str, dict[str, object]]] = []

    def __init__(self, **kwargs):
        _ = kwargs
        self.alive = False

    def start(self) -> None:
        self.alive = True
        type(self).starts += 1

    def close(self) -> None:
        self.alive = False
        type(self).closes += 1

    def is_alive(self) -> bool:
        return self.alive

    def initialize(self) -> None:
        type(self).initializes += 1

    def list_tools(self) -> list[dict[str, object]]:
        return [
            {"name": "new_page", "description": "open page", "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
            {"name": "evaluate_script", "description": "eval", "inputSchema": {"type": "object", "properties": {"function": {"type": "string"}}, "required": ["function"]}},
        ]

    def call_tool(self, *, name: str, arguments: dict[str, object]) -> dict[str, object]:
        type(self).calls.append((name, dict(arguments)))
        return {"ok": True, "name": name}


def test_mcp_call_reuses_native_session_across_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "ops")
    monkeypatch.setattr(mcp_call_tool_module, "_allowed_roots", lambda: [str(tmp_path)])
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp"],
                "operations": {},
                "metadata": {"native_tools": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))

    _FakeNativeClient.starts = 0
    _FakeNativeClient.closes = 0
    _FakeNativeClient.initializes = 0
    _FakeNativeClient.calls = []

    connector = McpConnector(
        terminal=_DummyTerminal(),
        native_client_factory=lambda **kwargs: _FakeNativeClient(**kwargs),
        session_ttl_seconds=3600,
    )
    tool = McpCallTool(connector=connector)

    first = tool.execute(
        profile="chrome",
        operation="new_page",
        arguments={"url": "https://example.com"},
        cwd=str(tmp_path),
    )
    second = tool.execute(
        profile="chrome",
        operation="evaluate_script",
        arguments={"function": "() => document.title"},
        cwd=str(tmp_path),
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert _FakeNativeClient.starts == 1
    assert _FakeNativeClient.initializes == 1
    assert len(_FakeNativeClient.calls) == 2
