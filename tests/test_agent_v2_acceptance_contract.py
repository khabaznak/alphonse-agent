from __future__ import annotations

from alphonse.agent_v2.core.intelligence.acceptance_contract import apply_amendment
from alphonse.agent_v2.core.intelligence.acceptance_contract import apply_status_patch
from alphonse.agent_v2.core.intelligence.acceptance_contract import contract_from_markdown
from alphonse.agent_v2.core.intelligence.acceptance_contract import definition_hash
from alphonse.agent_v2.core.intelligence.acceptance_contract import render_contract
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.io.outbox import _latest_tool_result_response


def test_status_review_cannot_rewrite_acceptance_criterion() -> None:
    contract = contract_from_markdown("1.- [ ] Identify the exact medicine")
    original_hash = definition_hash(contract)

    updated, rejected = apply_status_patch(
        contract,
        {
            "updates": [
                {
                    "criterion_id": "ac-1",
                    "statement": "Find anything vaguely related",
                    "status": "satisfied",
                    "evidence_refs": ["tool-call:db-search"],
                }
            ]
        },
        valid_evidence_refs={"tool-call:db-search"},
    )

    assert rejected == []
    assert updated["criteria"][0]["statement"] == "Identify the exact medicine"
    assert definition_hash(updated) == original_hash
    assert updated["criteria"][0]["status"] == "satisfied"


def test_status_review_rejects_unknown_criterion_and_unbacked_success() -> None:
    contract = contract_from_markdown("1.- [ ] Identify the exact medicine")

    updated, rejected = apply_status_patch(
        contract,
        {
            "updates": [
                {"criterion_id": "ac-404", "status": "satisfied", "evidence_refs": ["tool-call:db-search"]},
                {"criterion_id": "ac-1", "status": "satisfied", "evidence_refs": []},
            ]
        },
        valid_evidence_refs={"tool-call:db-search"},
    )

    assert updated["criteria"][0]["status"] == "pending"
    assert "unknown_criterion:ac-404" in rejected
    assert "invalid_evidence:ac-1" in rejected


def test_steering_amendment_adds_without_changing_existing_definition() -> None:
    contract = contract_from_markdown("1.- [ ] Read the household temperature")

    updated, rejected = apply_amendment(
        contract,
        {"operations": [{"operation": "add", "statement": "Use the LG client instead of Home Assistant"}]},
        source_message_id="steering-1",
    )

    assert rejected == []
    assert [item["statement"] for item in updated["criteria"]] == [
        "Read the household temperature",
        "Use the LG client instead of Home Assistant",
    ]
    assert updated["revision"] == 2
    assert updated["amendments"][0]["source_message_id"] == "steering-1"


def test_interactive_completion_requires_prepared_response() -> None:
    task = TaskState(check_verdict="wip", acceptance_criteria_md="1.- [x] Medicine identified")

    from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import act_node

    act_node(task)

    assert task.status == "running"
    assert task.metadata["pending_user_response"] is True
    assert task.metadata["act_route"] == "plan"


def test_bash_stdout_is_evidence_not_an_outbound_response() -> None:
    task = TaskState(acceptance_criteria_md="1.- [ ] Medicine identified")
    task.append_plan_call(
        {
            "id": "db-search",
            "tool_id": "native.bash",
            "tool_name": "bash",
            "arguments": {"command": "sqlite3 family.sqlite ..."},
            "internal_state": "Searching records.",
        }
    )
    task.record_plan_call_success("db-search", {"exit_code": 0, "stdout": "Bifebral", "stderr": ""})

    assert _latest_tool_result_response(task.to_dict()) == ""


def test_v3_prepared_response_is_projected_without_a_legacy_respond_tool_call() -> None:
    task = TaskState(
        metadata={
            "prepared_user_response": {
                "source": "v3_direct_response",
                "message": "¡Hola, Alex! Qué gusto saludarte.",
            }
        }
    )

    assert _latest_tool_result_response(task.to_dict()) == "¡Hola, Alex! Qué gusto saludarte."


def test_contract_markdown_is_a_derived_status_view() -> None:
    contract = contract_from_markdown("1.- [ ] First outcome\n2.- [x] Second outcome")

    assert render_contract(contract) == "1.- [ ] First outcome\n2.- [x] Second outcome"
