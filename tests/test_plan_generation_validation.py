from __future__ import annotations

from alphonse.agent.cortex.nodes.plan import route_after_plan


def test_route_after_plan_retries_when_flagged() -> None:
    assert route_after_plan({"plan_retry": True}) == "plan_node"


def test_route_after_plan_goes_to_apology_on_capability_gap() -> None:
    state = {"plans": [{"plan_type": "CAPABILITY_GAP"}]}
    assert route_after_plan(state) == "apology_node"


def test_route_after_plan_goes_to_respond_by_default() -> None:
    assert route_after_plan({"plan_retry": False, "plans": []}) == "respond_node"
