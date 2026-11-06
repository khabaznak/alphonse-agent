from __future__ import annotations

import json

import pytest

from alphonse.agent_v2.outcome_judgments import OUTCOME_QUESTIONS

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


@pytest.mark.parametrize("probability,expected", [(0.95, True), (0.5, False), (0.05, False), (True, False), (float("nan"), False), (float("inf"), False), (1.1, False), (-0.1, False), (None, False)])
def test_acknowledgement_review_uses_bounded_noul_and_requires_confident_valid_answer(probability, expected):
    def transport(url, api_key, payload, timeout):
        assert payload["state"]["work_state"].startswith("admitted")
        assert payload["state"]["acknowledgement"] == "Listo, ya lo hice."
        assert payload["questions"]["safe_intention"]["type"] == "noul"
        assert "before any work starts" in payload["questions"]["safe_intention"]["instructions"]
        return {"answers": {"safe_intention": {"type": "noul", "noul": probability}}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"), transport=transport)
    assert provider.review_task_acknowledgement(message="Update it", acknowledgement="Listo, ya lo hice.") is expected


def test_acknowledgement_review_never_checks_only_a_prefix():
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"),
        transport=lambda *args: pytest.fail("oversized acknowledgement must be suppressed"))
    assert provider.review_task_acknowledgement(message="Update it", acknowledgement="I'll check. " * 120 + "Done!") is False


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


def test_jev_act_requests_only_focused_nouls_and_preserves_probabilities() -> None:
    expected = {"effect_requested": .09, "effect_supported": .32, "effect_contradicted": .51,
                "reply_addresses_request": .87, "reply_grounded": .93}

    def transport(url, api_key, payload, timeout):
        assert set(payload["questions"]) == set(expected)
        assert {question["type"] for question in payload["questions"].values()} == {"noul"}
        assert payload["state"]["task_understanding"]["version"] == 1
        return {"answers": {key: {"type": "noul", "noul": value} for key, value in expected.items()}}

    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="fixture"), transport=transport)
    result = provider.judge_act(state={"task_understanding": {"version": 1}})
    assert result.answers == expected
    assert (result.no_threshold, result.yes_threshold) == (.2, .8)


@pytest.mark.parametrize("bad_answer", [None, {"type": "choice", "choice": "yes"}, *(
    {"type": "noul", "noul": value} for value in [True, "0.9", float("nan"), float("inf"), -1, 2]
)])
def test_jev_act_rejects_missing_malformed_and_out_of_range_judgments(bad_answer):
    from alphonse.agent_v2.system_one import ACT_QUESTIONS, SystemOneActAnswerError

    def transport(*args):
        answers = {key: {"type": "noul", "noul": .99} for key in ACT_QUESTIONS}
        if bad_answer is None:
            del answers["effect_supported"]
        else:
            answers["effect_supported"] = bad_answer
        return {"answers": answers}

    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="fixture"), transport=transport)
    with pytest.raises(SystemOneActAnswerError, match="effect_supported") as exc:
        provider.judge_act(state={})
    assert exc.value.invalid_question_ids == ("effect_supported",)
    assert exc.value.answers == {key: .99 for key in ACT_QUESTIONS if key != "effect_supported"}


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
            probability = 0.94 if "require reading a local resource" in question["instructions"] else 0.03
            answers[key] = {"type": "noul", "noul": probability}
        return {"answers": answers, "model": "jev-latest"}

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=transport,
    )
    tools = (
        ToolDescriptor("native.bash", "bash", ToolKind.NATIVE, description="Search project files and read or write files with shell commands."),
        ToolDescriptor("native.respond", "respond", ToolKind.NATIVE),
    )

    result = provider.select_plan_tools(
        goal="Find the treatment", phase={"phase_id": "p", "objective": "Locate record"},
        tools=tools,
    )
    provider.select_plan_tools(goal="Another goal", phase={"phase_id": "p2", "objective": "Another plan"}, tools=tools)

    assert result.selected_tool_ids == ("native.bash",)
    assert result.rejected_tool_ids == ("native.respond",)
    assert len(payloads[0]["questions"]) == len(tools) + 2
    assert payloads[0]["questions"] == payloads[1]["questions"]
    assert all(question["type"] == "noul" for question in payloads[0]["questions"].values())
    instructions = [question["instructions"] for question in payloads[0]["questions"].values()]
    assert 'reading a local resource' in instructions[0]
    assert 'Does the requester need an answer about `request.text`?' in instructions
    assert all(len(question['instructions']) < 150 for question in payloads[0]['questions'].values())
    serialized = str(payloads[0]["questions"])
    assert "Semantic tags:" not in serialized
    assert "Expected inputs:" not in serialized
    assert "Effect:" not in serialized


def test_jev_tool_selection_uses_configured_threshold_for_request_and_phase() -> None:
    def transport(url, api_key, payload, timeout):
        _ = url, api_key, timeout
        return {
            "answers": {
                key: {"type": "noul", "noul": 0.70}
                for key in payload["questions"]
            },
            "model": "jev-latest",
        }

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(
            enabled=True, api_key="secret", validated_at="now", tool_selection_threshold=0.65,
        ),
        transport=transport,
    )
    tools = (
        ToolDescriptor("native.bash", "Bash", ToolKind.NATIVE),
        ToolDescriptor("native.respond", "Respond", ToolKind.NATIVE),
    )

    request_selection = provider.curate_request_tools(
        goal="Update a record and tell me what changed",
        system_prompt="Plan the task.", session_history="", tools=tools,
    )
    phase_selection = provider.select_plan_tools(
        goal="Update a record and tell me what changed",
        phase={"phase_id": "p1", "objective": "Update and report"}, tools=tools,
    )

    assert request_selection.selected_tool_ids == ("native.bash", "native.respond")
    assert phase_selection.selected_tool_ids == ("native.bash", "native.respond")


def test_jev_request_curation_receives_plan_instructions_and_bounded_context() -> None:
    captured = []

    def transport(url, api_key, payload, timeout):
        _ = url, api_key, timeout
        captured.append(payload)
        return {
            "answers": {
                key: {"type": "noul", "noul": 0.93}
                for key in payload["questions"]
            },
            "model": "jev-latest",
        }

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=transport,
    )
    tools = (
        ToolDescriptor("artifact.sonos", "Sonos", ToolKind.ARTIFACT, "Check speaker availability."),
        ToolDescriptor("native.respond", "Respond", ToolKind.NATIVE),
    )

    result = provider.curate_request_tools(
        goal="Check Sonos speakers",
        system_prompt="Strategic Plan uses curated tool descriptors.",
        session_history="User previously configured Sonos.",
        project_context="This project controls the living-room Sonos system.",
        durable_memory="The user prefers music at low volume in the morning.",
        tools=tools,
    )

    assert result.selected_tool_ids == ("native.respond",)
    state = captured[0]['state']
    assert state['request']['text'] == 'Check Sonos speakers'
    assert state['conversation'][0]['text'] == 'User previously configured Sonos.'
    assert state['known_context'] == [
        {'kind': 'project', 'text': 'This project controls the living-room Sonos system.'},
        {'kind': 'memory', 'text': 'The user prefers music at low volume in the morning.'},
    ]
    assert 'plan_system_prompt' not in state and 'task_context' not in state
    assert 'Strategic Plan uses curated tool descriptors.' not in str(state)
    assert "Sonos" not in str(captured[0]["questions"])
    serialized_questions = str(captured[0]["questions"])
    assert "artifact.sonos" not in serialized_questions
    assert "Does this phase need to send a direct response" not in serialized_questions


def test_jev_triages_each_plan_message_with_bounded_relevance_score() -> None:
    payloads = []

    def transport(url, api_key, payload, timeout):
        _ = url, api_key, timeout
        payloads.append(payload)
        answers = {}
        for index, key in enumerate(payload["questions"]):
            level = (1, 0, 2)[index]
            probabilities = {str(value): 0.03 for value in range(3)}
            probabilities[str(level)] = 0.94
            answers[key] = {"type": "score", "score": float(level), "confidence": 0.94,
                            "probabilities": probabilities}
        return {"answers": answers, "model": "jev-latest"}

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"), transport=transport,
    )
    selected = provider.triage_plan_messages(
        task={"task_id": "t1", "goal": "Do X"},
        candidates=[
            {"message_id": "m1", "sender": "alex", "text": "B has context about X"},
            {"message_id": "m2", "sender": "alex", "text": "Independent request"},
            {"message_id": "m3", "sender": "alex", "text": "Consult B before X"},
        ],
    )

    assert selected == ("m1", "m3")
    assert list(payloads[0]["questions"]) == [f"message_relevance_{i}" for i in range(3)]
    assert payloads[0]["questions"]["message_relevance_0"]["type"] == "score"
    assert len(payloads[0]["questions"]["message_relevance_0"]["criteria"]) == 3


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


def test_jev_admission_follows_context_curation() -> None:
    payloads = []

    def transport(url, api_key, payload, timeout):
        _ = url, api_key, timeout
        payloads.append(payload)
        return {
            "answers": {
                "requires_task": {"type": "noul", "noul": 0.1},
                "context_relevant_0": {"type": "noul", "noul": 0.93},
                "context_relevant_1": {"type": "noul", "noul": 0.12},
            },
            "model": "jev-latest",
        }

    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"), transport=transport,
    )
    decision = provider.classify_task_admission(
        message="What should I prepare this week?",
        recent_conversation="The household is planning meals for the week.",
        context_candidates=(
            {"id": "global:household-norms", "title": "Household norms", "snippet": "Shared food preferences."},
            {"id": "project:old-plan", "title": "Old project", "snippet": "An unrelated historical plan."},
        ),
    )

    assert len(payloads) == 2
    assert set(payloads[0]["questions"]) == {"context_relevant_0", "context_relevant_1"}
    assert set(payloads[1]["questions"]) == {"requires_task"}
    assert "Shared food preferences." in str(payloads[1]["state"])
    assert "An unrelated historical plan." not in str(payloads[1]["state"])
    assert "context_candidates" in payloads[0]["state"]
    assert decision.selected_context_ids == ("global:household-norms",)
    assert decision.context_probabilities == {"global:household-norms": 0.93, "project:old-plan": 0.12}


def test_jev_native_registry_template_covers_every_out_of_box_tool() -> None:
    registry = _load_jev_native_tool_registry()
    assert set(registry) == {
        "native.respond",
        "native.integration_manage",
        "native.bash",
        "native.deliver_message",
        "native.send_attachment",
        "native.ask_question",
        "native.scheduled_task",
        "native.scheduled_task_delivery",
        "native.skill_install",
        "native.skill_maintain",
        "native.skill_read",
        "native.web_search",
        "native.web_fetch",
        "native.search_memory",
        "native.read_memory_source",
    }
    bash_question = registry["native.bash"]
    assert 'local database' in bash_question['criteria']['true']
    assert len(bash_question['additional_judgments']) == 2
    assert 'changing local files' in bash_question['additional_judgments'][0]['instructions']
    assert 'local command' in bash_question['additional_judgments'][1]['instructions']
    assert all(not key.startswith('artifact.') for key in registry)


def test_jev_tactical_review_distinguishes_operational_success_from_semantic_completion() -> None:
    provider = JevCriterionDecisionProvider(
        SystemOneSettings(enabled=True, api_key="secret", validated_at="now"),
        transport=_transport(support=0.91),
    )

    result = provider.evaluate_tactical_progress(
        goal="Find the solar record",
        phase={"phase_id": "p", "objective": "Locate record"},
        subgoal={"subgoal_id": "s", "objective": "Find authoritative record"},
        action={"tool_id": "native.bash", "status": "success", "result": {"stdout": "backlog.md"}},
    )

    assert result.complete is True
    assert result.confident is True


def test_intermediate_action_questions_cannot_complete_the_declared_subgoal():
    def transport(url, api_key, payload, timeout):
        assert payload["state"]["stage_goal"]["objective"] == "Read the live temperature"
        return {"answers": {key: {"type": "noul", "noul": (
            0.05 if key == "runtime_declared_subgoal_complete" else 0.95
        )} for key in payload["questions"]}}

    provider = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_key="fixture", validated_at="now"), transport=transport)
    result = provider.evaluate_tactical_progress(goal="Read temperature", phase={},
        subgoal={"objective": "Read the live temperature"},
        action={"status": "success", "result": {"stdout": "Usage: sensor read"}},
        questions=[{"question_id": "help_found", "type": "noul", "instructions": "Was CLI help found?",
                    "criteria": {"true": "Help found", "false": "Help missing"}}])
    assert not result.complete
    assert result.answers["help_found"] == 0.95
    assert result.answers["runtime_declared_subgoal_complete"] == 0.05


@pytest.mark.parametrize("stage_probability,expected_confidence,confident", [(.05, .95, True), (.5, .5, False), (.95, .95, True)])
def test_tactical_confidence_reports_weakest_individual_judgment(stage_probability, expected_confidence, confident):
    def transport(url, api_key, payload, timeout):
        return {"answers": {key: {"type": "noul", "noul": stage_probability if key == "runtime_declared_subgoal_complete" else .95}
                            for key in payload["questions"]}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_key="fixture"), transport=transport)
    result = provider.evaluate_tactical_progress(goal="Find source", phase={}, subgoal={}, action={"status": "success"})
    assert result.confidence == pytest.approx(expected_confidence)
    assert result.confident is confident
    assert result.to_metadata()["confidence_kind"] == "minimum_individual_noul_certainty"
    assert result.complete is (stage_probability == .95)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, -0.1, 1.1])
def test_malformed_tactical_probabilities_cannot_manufacture_completion(invalid):
    def transport(url, api_key, payload, timeout):
        return {"answers": {key: {"type": "noul", "noul": invalid} for key in payload["questions"]}}

    provider = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_key="fixture", validated_at="now"), transport=transport)
    with pytest.raises(ValueError, match="tactical_review_invalid"):
        provider.evaluate_tactical_progress(goal="Read live data", phase={}, subgoal={}, action={"status": "success"})


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
        "runtime_declared_subgoal_complete",
        "runtime_action_added_progress",
        "retry_fuse_transient_failure",
        "retry_fuse_same_call_likely_to_work",
        "retry_fuse_repeat_is_safe",
    }
    assert payload["state"]["tool_execution_log"][0]["error"] == "timeout"
    assert all(question["type"] == "noul" for question in payload["questions"].values())
    assert result.complete is True
    assert result.retry_approved is False


def test_memory_candidate_review_is_parallel_bounded_noul():
    seen = []
    def transport(url, key, payload, timeout):
        seen.append(payload)
        return {"answers": {qid: {"type": "noul", "noul": 0.95 if qid == "memory_0" else 0.05}
                            for qid in payload["questions"]}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"), transport=transport)
    assert provider.review_memory_candidates({"source": "bounded"}, [{"content": "one"}, {"content": "two"}]) == [True, False]
    assert len(seen) == 1
    assert all(q["type"] == "noul" for q in seen[0]["questions"].values())
    assert provider.review_memory_candidates({}, []) == []
    assert len(seen) == 1


@pytest.mark.parametrize("probability", [0.5, float("nan"), float("inf"), -1, 2, True, None])
def test_memory_review_ambiguous_or_malformed_answers_cannot_write(probability):
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"), transport=lambda *args: {
        "answers": {"memory_0": {"type": "noul", "noul": probability}}})
    with pytest.raises(ValueError):
        provider.review_memory_candidates({}, [{"content": "one"}])


def test_memory_semantic_comparison_uses_parallel_choices_not_wording():
    from types import SimpleNamespace
    from alphonse.agent_v2.core.memory.knowledge import KnowledgeRecord
    from alphonse.agent_v2.core.memory.contracts import AccessScope, SourceReference
    source = SourceReference("conversation:one", "conversation", AccessScope("alex"))
    old = KnowledgeRecord("memory:old", "My filter is FX-204.", AccessScope("alex"), (source,),
                          "2026-10-17T00:00:00+00:00", "fact", ("filter",), "filter.part", "user_report", ("filter",))
    captured = []
    def transport(url, key, payload, timeout):
        captured.append(payload)
        return {"answers": {qid: {"type": "choice", "choice": "equivalent", "probabilities": {"equivalent": 0.98, "uncertain": 0.02}}
                            for qid in payload["questions"]}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"), transport=transport)
    assert provider.compare_memory_claims([(old, old), (old, old)]) == ["equivalent", "equivalent"]
    assert len(captured) == 1
    assert all(q["type"] == "choice" and set(q["criteria"]) == {"equivalent", "compatible", "conflicting", "uncertain"}
               for q in captured[0]["questions"].values())
    assert "sources" not in captured[0]["state"][0]["old"]


@pytest.mark.parametrize("answer", [
    {"type": "choice", "choice": "uncertain", "probabilities": {"uncertain": 1}},
    {"type": "choice", "choice": "equivalent", "probabilities": {"equivalent": 0.5, "conflicting": 0.5}},
    {"type": "choice", "choice": "equivalent", "probabilities": {"equivalent": float("nan")}},
    {"type": "choice", "choice": "equivalent", "probabilities": {"equivalent": True}},
    {"type": "noul", "noul": 1},
])
def test_memory_comparison_ambiguity_or_invalid_answer_defers_write(answer):
    from types import SimpleNamespace
    fields = {key: "value" for key in ("memory_id", "content", "kind", "subjects", "claim_key", "basis", "observed_at", "valid_from", "valid_until", "persistence", "commitment_state")}
    record = SimpleNamespace(**fields)
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"), transport=lambda *args: {"answers": {"relation_0": answer}})
    with pytest.raises(ValueError):
        provider.compare_memory_claims([(record, record)])


def test_memory_support_selection_uses_noul_before_exposing_knowledge_to_extractor():
    from types import SimpleNamespace
    source = SimpleNamespace(to_dict=lambda: {"content": "Remember that number."})
    prior = [SimpleNamespace(memory_id="memory:one", to_dict=lambda: {"memory_id": "memory:one", "content": "Filter FX-204."})]
    captured = []
    def transport(url, key, payload, timeout):
        captured.append(payload)
        return {"answers": {"support_0": {"type": "noul", "noul": 0.98}}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"), transport=transport)
    assert provider.select_memory_support(source, prior) == ["memory:one"]
    assert captured[0]["questions"]["support_0"]["type"] == "noul"
    assert provider.select_memory_support(source, []) == []
    assert len(captured) == 1


@pytest.mark.parametrize("probability", [0.5, float("nan"), None, True])
def test_memory_support_selection_never_guesses_on_ambiguous_or_invalid_answers(probability):
    from types import SimpleNamespace
    item = SimpleNamespace(memory_id="memory:one", to_dict=lambda: {"content": "Filter"})
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"), transport=lambda *args: {
        "answers": {"support_0": {"type": "noul", "noul": probability}}})
    with pytest.raises(ValueError):
        provider.select_memory_support(item, [item])


@pytest.mark.parametrize("probability,expected", [(0.95, True), (0.5, False), (0.05, False), (True, False), (float("nan"), False), (1.1, False)])
def test_failure_wording_review_cannot_turn_failed_state_into_success(probability, expected):
    def transport(url, api_key, payload, timeout):
        assert payload["state"]["work_state"] == "failed; overall result unverified"
        assert payload["questions"]["honest_failure"]["type"] == "noul"
        assert "Reject reassuring overall success" in payload["questions"]["honest_failure"]["instructions"]
        return {"answers": {"honest_failure": {"type": "noul", "noul": probability}}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="secret"), transport=transport)
    assert provider.review_task_failure(message="Update it", explanation="Could not update: service offline. Try again later.") is expected


@pytest.mark.parametrize('code,attempts', [
    ('system_one_http_error:413', 1), ('system_one_http_error:400', 1),
    ('unexpected failure with PRIVATE_PROVIDER_BODY', 1),
    ('system_one_temporarily_unavailable:529', 4), ('system_one_http_error:503', 4),
    ('system_one_connection_failed', 4),
])
def test_provider_failure_diagnostics_are_safe_and_only_transient_errors_retry(monkeypatch, code, attempts):
    from alphonse.agent_v2 import system_one
    from alphonse.agent_v2.core.context_manifests import manifest_scope
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    calls = []
    sleeps = []
    monkeypatch.setattr(system_one, 'sleep', sleeps.append)

    def transport(*args):
        calls.append(True)
        raise RuntimeError(code)

    task = TaskState(user='fixture', task_id='diagnostics')
    context = CoreLoopContext(messages=InMemoryMessageQueue())
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='PRIVATE_API_KEY'), transport=transport)
    with manifest_scope(task, context), pytest.raises(RuntimeError):
        provider.judge_act(state={'goal': 'Inspect fixture'})
    assert len(calls) == attempts
    assert len(sleeps) == attempts - 1
    manifest = task.metadata['decision_context_manifests'][-1]
    assert manifest['status'] == 'failed'
    assert manifest['exception_type'] == 'RuntimeError'
    assert manifest['error_code'] == ('unclassified' if 'PRIVATE_PROVIDER_BODY' in code else code)
    assert 'PRIVATE_PROVIDER_BODY' not in str(task.metadata)
    assert 'PRIVATE_API_KEY' not in str(task.metadata)


@pytest.mark.parametrize('code,expected', [('max_tokens_exceeded', 'system_one_context_limit_exceeded'),
                                         ('unrecognized_PRIVATE_BODY', 'system_one_request_rejected:400')])
def test_provider_request_rejection_retains_machine_diagnostics_without_body_or_retry(monkeypatch, code, expected):
    import io
    from urllib.error import HTTPError
    from alphonse.agent_v2 import system_one
    from alphonse.agent_v2.core.context_manifests import manifest_scope
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    calls = []
    def reject(*args, **kwargs):
        calls.append(True)
        body = json.dumps({'error': {'code': code, 'message': 'PRIVATE_MEDICAL_TEXT PRIVATE_API_KEY'}}).encode()
        raise HTTPError('https://api.typesafe.ai/v1/systemone', 400, 'Bad Request', {}, io.BytesIO(body))
    monkeypatch.setattr(system_one, 'urlopen', reject)
    monkeypatch.setattr(system_one, 'sleep', lambda *args: pytest.fail('Rejected requests must not retry'))
    task = TaskState(task_id='synthetic', user='fixture')
    context = CoreLoopContext(messages=InMemoryMessageQueue())
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='PRIVATE_API_KEY'))
    with manifest_scope(task, context), pytest.raises(system_one.SystemOneRequestError) as error:
        provider.judge_act(state={'goal': 'Read a fixture record'})
    assert str(error.value) == expected and len(calls) == 1
    manifest = task.metadata['decision_context_manifests'][-1]
    assert manifest['error_code'] == expected
    assert manifest['provider_error']['http_status'] == 400
    serialized = json.dumps(manifest)
    assert 'PRIVATE_MEDICAL_TEXT' not in serialized and 'PRIVATE_API_KEY' not in serialized
    assert 'unrecognized_PRIVATE_BODY' not in serialized


def test_act_request_budget_includes_questions_and_stops_before_transport():
    from alphonse.agent_v2 import system_one
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'),
        transport=lambda *args: pytest.fail('Oversized request must not reach transport'))
    with pytest.raises(ValueError, match='system_one_context_budget_exceeded'):
        provider.judge_act(state={'goal': 'Large fixture ' * 4000})
