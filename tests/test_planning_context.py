from alphonse.agent.cognition.planning import PlanningMode, resolve_planning_context
from alphonse.agent.cortex import utils as cortex_utils


def test_planning_context_respects_cli_override() -> None:
    context = resolve_planning_context(
        autonomy_level=0.2,
        requested_mode=PlanningMode.AVENTURIZACION,
    )
    assert context.user_override is True
    assert context.effective_mode == PlanningMode.AVENTURIZACION
    assert context.requested_mode == PlanningMode.AVENTURIZACION
    assert context.autonomy_level == 0.2


def test_cortex_state_persists_planning_fields() -> None:
    state = {
        "autonomy_level": 0.8,
        "planning_mode": "aventurizacion",
    }
    cognition_state = cortex_utils.build_cognition_state(state)
    meta = cortex_utils.build_meta(state)
    assert cognition_state["autonomy_level"] == 0.8
    assert cognition_state["planning_mode"] == "aventurizacion"
    assert meta["autonomy_level"] == 0.8
    assert meta["planning_mode"] == "aventurizacion"
