from __future__ import annotations

import pytest

from alphonse.integrations.homeassistant import config as ha_config


def test_load_homeassistant_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HA_BASE_URL", "http://homeassistant.local:8123")
    monkeypatch.setenv("HA_TOKEN", "abc123")
    monkeypatch.setenv("HA_ALLOWED_DOMAINS", "light,sensor")
    monkeypatch.setenv("HA_ALLOWED_ENTITY_IDS", "light.kitchen")
    monkeypatch.setenv("HA_DEBOUNCE_ENABLED", "true")
    monkeypatch.setattr(ha_config, "get_active_tool_config", lambda _tool_key: None)

    loaded = ha_config.load_homeassistant_config()

    assert loaded is not None
    assert loaded.base_url == "http://homeassistant.local:8123"
    assert loaded.token == "abc123"
    assert loaded.allowed_domains == frozenset({"light", "sensor"})
    assert loaded.allowed_entity_ids == frozenset({"light.kitchen"})
    assert loaded.debounce.enabled is True


def test_load_homeassistant_config_tool_config_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HA_BASE_URL", "http://env.local:8123")
    monkeypatch.setenv("HA_TOKEN", "env-token")
    monkeypatch.setattr(
        ha_config,
        "get_active_tool_config",
        lambda _tool_key: {
            "config": {
                "HA_BASE_URL": "http://db.local:8123",
                "HA_TOKEN": "db-token",
                "HA_DEBOUNCE_KEY_STRATEGY": "entity",
            }
        },
    )

    loaded = ha_config.load_homeassistant_config()

    assert loaded is not None
    assert loaded.base_url == "http://db.local:8123"
    assert loaded.token == "db-token"
    assert loaded.debounce.key_strategy == "entity"


def test_load_homeassistant_config_missing_required_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HA_BASE_URL", raising=False)
    monkeypatch.delenv("HA_TOKEN", raising=False)
    monkeypatch.setattr(ha_config, "get_active_tool_config", lambda _tool_key: None)

    loaded = ha_config.load_homeassistant_config()

    assert loaded is None
