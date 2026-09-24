from __future__ import annotations

import pytest

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider
from alphonse.agent_v2.system_one import SQLiteSystemOneSettingsStore
from alphonse.agent_v2.system_one import SystemOneSettings
from alphonse.agent_v2.system_one import TypeSafeSystemOneClient
from alphonse.agent_v2.system_one import _load_jev_native_tool_registry
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


def test_jev_maps_only_direct_evidence_refs_during_check_review() -> None:
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
    assert result.recommended_route == ""
    assert result.ambiguous_criterion_ids == ()


def test_jev_act_recommendation_combines_choice_with_parallel_resilience_fuses() -> None:
    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=_transport(route="ask_user", support=0.91),
    )

    result = provider.recommend_act(state={"check_verdict": "failure", "goal": "Connect to server"})

    assert result.action == "ask_user"
    assert result.confident is True
    assert set(result.answers) == {
        "continuation_is_worthwhile", "user_input_can_unblock", "closure_explanation_is_warranted",
    }


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
            probability = 0.94 if "locate text or records" in question["instructions"] else 0.03
            answers[key] = {"type": "noul", "noul": probability}
        return {"answers": answers, "model": "jev-latest"}

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=transport,
    )
    tools = (
        ToolDescriptor("native.project_search", "project_search", ToolKind.NATIVE),
        ToolDescriptor("native.respond", "respond", ToolKind.NATIVE),
    )

    result = provider.select_plan_tools(
        goal="Find the treatment", phase={"phase_id": "p", "objective": "Locate record"},
        tools=tools,
    )
    provider.select_plan_tools(goal="Another goal", phase={"phase_id": "p2", "objective": "Another plan"}, tools=tools)

    assert result.selected_tool_ids == ("native.project_search",)
    assert result.rejected_tool_ids == ("native.respond",)
    assert len(payloads[0]["questions"]) == len(tools)
    assert payloads[0]["questions"] == payloads[1]["questions"]
    assert all(question["type"] == "noul" for question in payloads[0]["questions"].values())
    instructions = [question["instructions"] for question in payloads[0]["questions"].values()]
    assert "Does this phase need to locate text or records inside the authorized project?" in instructions
    assert "Does this phase need to send a direct response to the requester?" in instructions
    questions = list(payloads[0]["questions"].values())
    assert questions[0]["criteria"] == {
        "true": "The phase needs to discover which project file contains a relevant term, record, or reference before reading or changing it.",
        "false": "The relevant file is already known, the information is outside the project, or no project-file discovery is needed.",
    }
    assert questions[1]["criteria"] == {
        "true": "The phase includes a user-visible answer, greeting, status update, clarification, or presentation of completed work.",
        "false": "The phase must perform or verify other work before responding, or it has no requester-facing response subgoal.",
    }
    serialized = str(payloads[0]["questions"])
    assert "Semantic tags:" not in serialized
    assert "Expected inputs:" not in serialized
    assert "Effect:" not in serialized


def test_jev_triages_each_plan_message_with_choice_and_relevance_fuse() -> None:
    payloads = []

    def transport(url, api_key, payload, timeout):
        _ = url, api_key, timeout
        payloads.append(payload)
        answers = {}
        for key, question in payload["questions"].items():
            if question["type"] == "choice":
                answers[key] = {"type": "choice", "choice": "relevant_context", "probabilities": {"relevant_context": 0.91}}
            else:
                answers[key] = {"type": "noul", "noul": 0.94}
        return {"answers": answers, "model": "jev-latest"}

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"), transport=transport,
    )
    selected = provider.triage_plan_messages(
        task={"task_id": "t1", "goal": "Do X"},
        candidates=[{"message_id": "m1", "sender": "alex", "text": "Consult B first"}],
    )

    assert selected == ("m1",)
    assert list(payloads[0]["questions"]) == ["message_class_0", "relevant_0"]
    assert payloads[0]["questions"]["message_class_0"]["type"] == "choice"
    assert payloads[0]["questions"]["relevant_0"]["type"] == "noul"


def test_jev_admission_classifies_work_beyond_one_direct_text_reply() -> None:
    payloads = []

    def transport(url, api_key, payload, timeout):
        _ = url, api_key, timeout
        payloads.append(payload)
        return {
            "answers": {"requires_task": {"type": "noul", "noul": 0.93}},
            "model": "jev-latest",
        }

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"), transport=transport,
    )

    decision = provider.classify_task_admission(message="Inspect the project and repair the failing test")

    assert decision.requires_task is True
    assert decision.confident is True
    assert payloads[0]["questions"]["requires_task"]["type"] == "noul"
    criteria = payloads[0]["questions"]["requires_task"]["criteria"]
    assert "planning" in criteria["true"]
    assert "One immediate conversational text reply" in criteria["false"]


def test_jev_native_registry_template_covers_every_out_of_box_tool() -> None:
    assert set(_load_jev_native_tool_registry()) == {
        "native.respond",
        "native.bash",
        "native.exact_text_edit",
        "native.project_search",
        "native.read_project_file",
        "native.deliver_message",
        "native.send_attachment",
        "native.ask_question",
        "native.scheduled_task",
        "native.artifact_registration",
        "native.analyze_image",
        "native.web_search",
        "native.web_fetch",
    }


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


def test_jev_tactical_review_sends_acceptance_and_retry_fuse_questions_together() -> None:
    payloads = []

    def transport(url, api_key, payload, timeout):
        _ = url, api_key, timeout
        payloads.append(payload)
        return {
            "answers": {
                key: {
                    "type": "noul",
                    "noul": 0.1 if key == "retry_fuse_repeat_is_safe" else 0.95,
                }
                for key in payload["questions"]
            },
            "model": "jev-latest",
        }

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=transport,
    )
    result = provider.evaluate_tactical_progress(
        goal="Read from server",
        phase={"phase_id": "p", "objective": "Read record"},
        subgoal={"subgoal_id": "s", "objective": "Read record"},
        action={"tool_id": "native.server_read", "status": "failed", "error": "timeout"},
        questions=[{
            "question_id": "record_was_read",
            "type": "noul",
            "instructions": "Was the record read?",
            "criteria": {"true": "The record is present.", "false": "The record is absent."},
        }],
        execution_log=[{"tool_id": "native.server_read", "status": "failed", "error": "timeout"}],
    )

    payload = payloads[0]
    assert set(payload["questions"]) == {
        "record_was_read",
        "retry_fuse_transient_failure",
        "retry_fuse_same_call_likely_to_work",
        "retry_fuse_repeat_is_safe",
    }
    assert payload["state"]["tool_execution_log"][0]["error"] == "timeout"
    assert all(question["type"] == "noul" for question in payload["questions"].values())
    assert result.complete is True
    assert result.retry_approved is False
