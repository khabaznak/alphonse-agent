from __future__ import annotations

from alphonse.agent_v2.integrations import SQLiteIntegrationStore
from alphonse.agent_v2.integrations import build_default_integration_registry


def test_integration_store_upserts_lists_and_masks_secrets() -> None:
    store = SQLiteIntegrationStore()

    record = store.upsert(
        integration_id="telegram-home",
        provider_key="telegram",
        display_name="Telegram Home",
        enabled=True,
        config={"poll_interval_sec": 2.0},
        secrets={"bot_token": "secret-token"},
    )

    assert record.integration_id == "telegram-home"
    assert store.get("telegram-home") == record
    assert store.list_enabled() == [record]
    assert record.to_dict()["secrets"] == {"bot_token": "***"}
    assert record.to_dict(mask_secrets=False)["secrets"] == {"bot_token": "secret-token"}


def test_integration_store_can_disable_and_remove_token() -> None:
    store = SQLiteIntegrationStore()
    store.upsert(
        integration_id="telegram-home",
        provider_key="telegram",
        display_name="Telegram Home",
        enabled=True,
        secrets={"bot_token": "secret-token"},
    )

    disabled = store.set_enabled("telegram-home", False)
    without_token = store.remove_secret("telegram-home", "bot_token")

    assert disabled.enabled is False
    assert store.list_enabled() == []
    assert without_token.secrets == {}


def test_default_integration_registry_exposes_telegram_descriptor() -> None:
    registry = build_default_integration_registry()

    descriptor = registry.get("telegram")

    assert descriptor is not None
    assert descriptor.default_integration_id == "telegram-home"
    assert descriptor.runtime_factory is not None
