from __future__ import annotations

import json
from pathlib import Path

from alphonse.agent.tools.mcp.registry import McpProfileRegistry


def test_mcp_registry_loads_profiles_from_json(monkeypatch, tmp_path: Path) -> None:
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
                        "description": "search",
                        "command_template": "search {query}",
                        "required_args": ["query"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))

    registry = McpProfileRegistry()
    profile = registry.get("chrome")
    assert profile is not None
    assert "web_search" in profile.operations


def test_mcp_registry_allows_native_profile_without_operations(monkeypatch, tmp_path: Path) -> None:
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

    registry = McpProfileRegistry()
    profile = registry.get("chrome")
    assert profile is not None
    assert profile.operations == {}
