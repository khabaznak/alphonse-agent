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
                    "web_search": {
                        "key": "web_search",
                        "description": "search web",
                        "command_template": "search {query}",
                        "required_args": ["query"],
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
        operation="web_search",
        arguments={"query": "Veloswim company profile"},
        cwd=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert dummy_terminal.calls
    call = dummy_terminal.calls[0]
    command = str(call.get("command") or "")
    assert "chrome-devtools-mcp" in command or "chrome-mcp" in command or "npx -y chrome-devtools-mcp" in command
    metadata = result.get("metadata")
    assert isinstance(metadata, dict)
    envelope = metadata.get("policy_envelope")
    assert isinstance(envelope, dict)
    assert envelope.get("execution_surface") == "mcp"
    assert envelope.get("profile") == "chrome"
    assert envelope.get("operation") == "web_search"


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
        query="Veloswim",
        mode="headless",
        cwd=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert dummy_terminal.calls
    command = str(dummy_terminal.calls[0].get("command") or "")
    assert "Veloswim" in command


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
        args={
            "profile": "chrome",
            "operation": "web_search",
            "query": "Veloswim nested",
        },
        cwd=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert dummy_terminal.calls
    command = str(dummy_terminal.calls[0].get("command") or "")
    assert "Veloswim nested" in command
