from __future__ import annotations

import json

import pytest

from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.inference_settings import CODEX_DEFAULT_MODEL
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord
from alphonse.agent_v2.inference_settings import SQLiteInferenceSettingsStore
from alphonse.agent_v2.inference_settings import build_inference_router_from_settings
from alphonse.agent_v2.inference_settings import provider_status
from alphonse.agent_v2.inference_settings import validate_and_save_inference_settings
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.runtime import refresh_runtime_inference


def test_inference_settings_uses_environment_only_until_a_selection_is_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_CODEX_MODEL", "env-model")
    store = SQLiteInferenceSettingsStore(":memory:")

    assert store.get().model_id == "env-model"

    saved = store.save(InferenceSettingsRecord(model_id="saved-model"))

    assert saved.model_id == "saved-model"
    assert store.get().model_id == "saved-model"


def test_codex_catalog_filters_hidden_models_and_keeps_default(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / "models_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "fetched_at": "2026-07-11T00:00:00Z",
                "client_version": "0.142.5",
                "models": [
                    {"slug": "gpt-5.5", "display_name": "GPT-5.5", "visibility": "list"},
                    {"slug": "hidden", "visibility": "hidden"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_V2_CODEX_MODELS_CACHE_PATH", str(cache_path))
    monkeypatch.setattr("alphonse.agent_v2.inference_settings._codex_cli_version", lambda: "codex 0.142.5")

    status = provider_status("openai_codex")

    assert [item["model_id"] for item in status["models"]] == [CODEX_DEFAULT_MODEL, "gpt-5.5"]
    assert status["catalog_fetched_at"] == "2026-07-11T00:00:00Z"


def test_validation_failure_does_not_replace_active_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteInferenceSettingsStore(":memory:")
    store.save(InferenceSettingsRecord(model_id="working"))
    monkeypatch.setattr(
        "alphonse.agent_v2.inference_settings.InferenceProviderDescriptor.validate",
        lambda self, model_id: (_ for _ in ()).throw(ValueError("openai_codex_cli_upgrade_required")),
    )

    with pytest.raises(ValueError, match="openai_codex_cli_upgrade_required"):
        validate_and_save_inference_settings(store, provider_key="openai_codex", model_id="broken")

    assert store.get().model_id == "working"
    assert store.get().validation_error == "openai_codex_cli_upgrade_required"


def test_saved_selection_replaces_router_for_future_tasks_only() -> None:
    runtime = build_runtime_host(
        inference=InferenceRouter(provider=StubInferenceProvider(), default_profile=build_inference_router_from_settings(InferenceSettingsRecord(model_id="old")).default_profile),
        inference_settings_store=SQLiteInferenceSettingsStore(":memory:"),
    )
    active_context_router = runtime.core.inference
    settings = runtime.inference_settings_store.save(InferenceSettingsRecord(model_id="new"))

    refresh_runtime_inference(runtime, settings)

    assert active_context_router is not runtime.core.inference
    assert active_context_router.default_profile.model == "old"
    assert runtime.core.inference is not None
    assert runtime.core.inference.default_profile.model == "new"
