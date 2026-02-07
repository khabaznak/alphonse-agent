from __future__ import annotations

from pathlib import Path

from alphonse.agent.cognition.intent_catalog import IntentCatalogService, IntentCatalogStore, IntentSpec
from alphonse.agent.cognition.intent_detector_llm import IntentDetectorLLM
from alphonse.agent.nervous_system.migrate import apply_schema


def _setup_db(tmp_path: Path) -> IntentCatalogService:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    store = IntentCatalogStore(str(db_path))
    return IntentCatalogService(store=store, ttl_seconds=0)


def test_disabled_intent_not_selected(tmp_path: Path) -> None:
    service = _setup_db(tmp_path)
    store = service.store
    disabled = IntentSpec(
        intent_name="custom.disabled",
        category="core_conversational",
        description="Disabled intent",
        examples=["DisabledExample"],
        required_slots=[],
        optional_slots=[],
        default_mode="aventurizacion",
        risk_level="low",
        handler="custom.disabled",
        enabled=False,
        intent_version="1.0.0",
        origin="factory",
    )
    store.upsert(disabled)
    detector = IntentDetectorLLM(service)
    result = detector.detect("DisabledExample", llm_client=None)
    assert result is None


def test_enabled_intent_matches_examples(tmp_path: Path) -> None:
    service = _setup_db(tmp_path)
    store = service.store
    enabled = IntentSpec(
        intent_name="custom.enabled",
        category="core_conversational",
        description="Enabled intent",
        examples=["EnabledExample"],
        required_slots=[],
        optional_slots=[],
        default_mode="aventurizacion",
        risk_level="low",
        handler="custom.enabled",
        enabled=True,
        intent_version="1.0.0",
        origin="factory",
    )
    store.upsert(enabled)
    detector = IntentDetectorLLM(service)
    result = detector.detect("EnabledExample", llm_client=None)
    assert result is not None
    assert result.intent_name == "custom.enabled"


class ExplodingLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[no-untyped-def]
        raise AssertionError("LLM should not be called for fast path")


def test_fast_path_skips_llm(tmp_path: Path) -> None:
    service = _setup_db(tmp_path)
    detector = IntentDetectorLLM(service)
    result = detector.detect("Hi Alphonse", llm_client=ExplodingLLM())
    assert result is not None
    assert result.intent_name == "greeting"
