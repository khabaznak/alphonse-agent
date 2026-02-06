from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.prompt_store import PromptContext, SqlitePromptStore
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
