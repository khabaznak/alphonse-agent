from __future__ import annotations

import pytest

from alphonse.agent_v2.system_one import JevCriterionDecisionProvider
from alphonse.agent_v2.system_one import SQLiteSystemOneSettingsStore
from alphonse.agent_v2.system_one import SystemOneSettings
from alphonse.agent_v2.system_one import TypeSafeSystemOneClient
from alphonse.agent_v2.system_one import build_system_one_provider
from alphonse.agent_v2.system_one import validate_and_save_system_one_settings


def _transport(route="continue", support=0.91, route_probability=0.84):
    def send(url, api_key, payload, timeout):
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert api_key == "secret"
        assert timeout >= 1
        answers = {}
        for key, question in payload["questions"].items():
            if question["type"] == "choice":
                answers[key] = {
                    "type": "choice", "choice": route, "confidence": route_probability,
                    "probabilities": {route: route_probability},
                }
            else:
                answers[key] = {"type": "noul", "noul": support}
        return {"answers": answers, "model": "jev-latest", "usage": {"tokens": 17}}
    return send


def test_system_one_settings_mask_key_and_require_validation_before_runtime_use(tmp_path) -> None:
    store = SQLiteSystemOneSettingsStore(tmp_path / "settings.sqlite3")
    saved = validate_and_save_system_one_settings(
        store,
        values={"enabled": True, "api_key": "secret"},
        transport=_transport(),
    )

    assert saved.validated_at
    assert saved.to_dict()["has_api_key"] is True
    assert "api_key" not in saved.to_dict()
    assert build_system_one_provider(saved) is not None


def test_system_one_validation_preserves_saved_settings_when_new_key_fails(tmp_path) -> None:
    store = SQLiteSystemOneSettingsStore(tmp_path / "settings.sqlite3")
    original = validate_and_save_system_one_settings(
        store, values={"enabled": True, "api_key": "secret"}, transport=_transport(),
    )

    with pytest.raises(ValueError, match="bad_key"):
        validate_and_save_system_one_settings(
            store,
            values={"enabled": True, "api_key": "replacement"},
            transport=lambda *_args: (_ for _ in ()).throw(ValueError("bad_key")),
        )

    assert store.get() == original


def test_jev_maps_only_direct_evidence_refs_and_returns_act_recommendation() -> None:
    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=_transport(route="replan"),
    )
    result = provider.evaluate(
        contract={"criteria": [{"id": "ac-1", "statement": "Record is updated", "required": True, "status": "pending"}]},
        phase={"phase_id": "phase-1", "objective": "Update record"},
        evidence={"entries": [{"evidence_ref": "action:1", "status": "success", "result": {"updated": True}}]},
    )

    assert result.updates[0]["status"] == "satisfied"
    assert result.updates[0]["evidence_refs"] == ["action:1"]
    assert result.recommended_route == "replan"
    assert result.route_confident is True
    assert result.ambiguous_criterion_ids == ()


def test_jev_marks_midrange_evidence_ambiguous_for_system_two_fallback() -> None:
    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=_transport(support=0.51),
    )
    result = provider.evaluate(
        contract={"criteria": [{"id": "ac-1", "statement": "Record is updated"}]},
        phase={"phase_id": "phase-1", "objective": "Update record"},
        evidence={"entries": [{"evidence_ref": "action:1", "status": "success", "result": {"updated": "maybe"}}]},
    )

    assert result.ambiguous_criterion_ids == ("ac-1",)
    assert result.updates == ()


def test_system_one_client_rejects_missing_key_before_transport() -> None:
    client = TypeSafeSystemOneClient(
        api_url="https://api.typesafe.ai/v1/systemone", api_key="", model="jev-latest",
        transport=lambda *_args: pytest.fail("transport must not run"),
    )
    with pytest.raises(ValueError, match="system_one_api_key_required"):
        client.validate()


def test_jev_classifies_complete_static_tool_registry_with_parallel_noul_questions() -> None:
    payloads = []

    def transport(url, api_key, payload, timeout):
        _ = url, api_key, timeout
        payloads.append(payload)
        answers = {}
        for key, question in payload["questions"].items():
            probability = 0.94 if "project_search" in question["instructions"] else 0.03
            answers[key] = {"type": "noul", "noul": probability}
        return {"answers": answers, "model": "jev-latest"}

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=transport,
    )
    tools = (
        type("Tool", (), {"tool_id": "native.project_search", "name": "search", "description": "Find project records", "read_only": True})(),
        type("Tool", (), {"tool_id": "artifact.medical", "name": "medical", "description": "Query medical records", "read_only": True})(),
    )

    result = provider.select_plan_tools(
        goal="Find the treatment", phase={"phase_id": "p", "objective": "Locate record"},
        tools=tools,
    )
    provider.select_plan_tools(goal="Another goal", phase={"phase_id": "p2", "objective": "Another plan"}, tools=tools)

    assert result.selected_tool_ids == ("native.project_search",)
    assert result.rejected_tool_ids == ("artifact.medical",)
    assert len(payloads[0]["questions"]) == len(tools)
    assert payloads[0]["questions"] == payloads[1]["questions"]
    assert all(question["type"] == "noul" for question in payloads[0]["questions"].values())


def test_jev_tactical_review_distinguishes_operational_success_from_semantic_completion() -> None:
    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=_transport(support=0.91),
    )

    result = provider.evaluate_tactical_progress(
        goal="Find the solar record",
        phase={"phase_id": "p", "objective": "Locate record"},
        subgoal={"subgoal_id": "s", "objective": "Find authoritative record"},
        action={"tool_id": "native.project_search", "status": "success", "result": {"path": "backlog.md"}},
    )

    assert result.complete is True
    assert result.confident is True
