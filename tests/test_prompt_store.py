from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.prompt_store import (
    NullPromptStore,
    PromptContext,
    SqlitePromptStore,
    seed_default_templates,
)
from alphonse.agent.cognition.response_composer import ResponseComposer
from alphonse.agent.cognition.response_spec import ResponseSpec
from alphonse.agent.nervous_system.migrate import apply_schema


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SqlitePromptStore:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    return SqlitePromptStore(str(db_path))


def test_best_match_prefers_exact_locale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _prepare_db(tmp_path, monkeypatch)
    store.upsert_template(
        key="clarify.intent",
        locale="es",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="ES general",
        enabled=True,
        priority=0,
        changed_by="test",
        reason="seed",
    )
    store.upsert_template(
        key="clarify.intent",
        locale="es-MX",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="ES MX",
        enabled=True,
        priority=0,
        changed_by="test",
        reason="seed",
    )
    match = store.get_template(
        "clarify.intent",
        PromptContext(locale="es-MX", address_style="tu", tone="casual"),
    )
    assert match is not None
    assert match.template == "ES MX"


def test_address_style_prefers_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _prepare_db(tmp_path, monkeypatch)
    store.upsert_template(
        key="core.identity.agent",
        locale="es",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="neutral",
        enabled=True,
        priority=0,
        changed_by="test",
        reason="seed",
    )
    store.upsert_template(
        key="core.identity.agent",
        locale="es",
        address_style="usted",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="formal",
        enabled=True,
        priority=1,
        changed_by="test",
        reason="seed",
    )
    match = store.get_template(
        "core.identity.agent",
        PromptContext(locale="es-MX", address_style="usted", tone="formal"),
    )
    assert match is not None
    assert match.template == "formal"


def test_rollback_restores_previous_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_db(tmp_path, monkeypatch)
    template_id = store.upsert_template(
        key="core.greeting.test",
        locale="en",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="Hello there",
        enabled=True,
        priority=5,
        changed_by="test",
        reason="v1",
    )
    store.upsert_template(
        key="core.greeting.test",
        locale="en",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="Updated greeting",
        enabled=True,
        priority=5,
        changed_by="test",
        reason="v2",
    )
    store.rollback_template(template_id, 1, changed_by="test", reason="rollback")
    match = store.get_template(
        "core.greeting.test",
        PromptContext(locale="en-US", address_style="tu", tone="casual"),
    )
    assert match is not None
    assert match.template == "Hello there"


def test_response_composer_prefers_prompt_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_db(tmp_path, monkeypatch)
    store.upsert_template(
        key="clarify.intent",
        locale="en",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="Custom clarify",
        enabled=True,
        priority=5,
        changed_by="test",
        reason="override",
    )
    composer = ResponseComposer(prompt_store=store)
    spec = ResponseSpec(
        kind="clarify",
        key="clarify.intent",
        locale="en-US",
        address_style="tu",
        tone="casual",
        channel="telegram",
        variant="default",
        policy_tier="safe",
    )
    assert composer.compose(spec) == "Custom clarify"


def test_seed_includes_executor_response_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_db(tmp_path, monkeypatch)
    seed_default_templates(str(tmp_path / "nerve-db"))
    keys = [
        "lan.device.not_found",
        "lan.device.armed",
        "pairing.not_found",
        "error.execution_failed",
        "policy.reminder.restricted",
    ]
    for key in keys:
        match = store.get_template(
            key,
            PromptContext(locale="es-MX", address_style="tu", tone="friendly"),
        )
        assert match is not None, key


def test_sensitive_response_uses_safe_fallback_when_prompt_missing() -> None:
    composer = ResponseComposer(prompt_store=NullPromptStore())
    spec = ResponseSpec(
        kind="policy_block",
        key="policy.reminder.restricted",
        locale="es-MX",
    )
    assert composer.compose(spec) == "policy.reminder.restricted"


def test_unknown_sensitive_key_uses_default_safe_fallback() -> None:
    composer = ResponseComposer(prompt_store=NullPromptStore())
    spec = ResponseSpec(
        kind="policy_block",
        key="policy.unknown",
        locale="en-US",
    )
    assert composer.compose(spec) == "policy.unknown"


def test_relaxed_matching_allows_selector_mismatch_for_non_sensitive_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_db(tmp_path, monkeypatch)
    store.upsert_template(
        key="core.greeting.custom_relaxed",
        locale="en",
        address_style="tu",
        tone="formal",
        channel="telegram",
        variant="default",
        policy_tier="safe",
        template="Hello from strict row",
        enabled=True,
        priority=0,
        changed_by="test",
        reason="seed",
    )
    match = store.get_template(
        "core.greeting.custom_relaxed",
        PromptContext(
            locale="en-US",
            address_style="usted",
            tone="casual",
            channel="cli",
            variant="default",
            policy_tier="safe",
        ),
    )
    assert match is not None
    assert match.template == "Hello from strict row"


def test_sensitive_keys_remain_strict_on_selector_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_db(tmp_path, monkeypatch)
    store.upsert_template(
        key="policy.custom.strict_test",
        locale="en",
        address_style="any",
        tone="formal",
        channel="any",
        variant="default",
        policy_tier="strict",
        template="STRICT POLICY RESPONSE",
        enabled=True,
        priority=0,
        changed_by="test",
        reason="seed",
    )
    match = store.get_template(
        "policy.custom.strict_test",
        PromptContext(
            locale="en-US",
            address_style="tu",
            tone="casual",
            channel="telegram",
            variant="default",
            policy_tier="strict",
        ),
    )
    assert match is None
