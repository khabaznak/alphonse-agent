from alphonse.agent.cognition.pending_interaction import (
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
)
from alphonse.agent.cortex.nodes import execution_helpers
from alphonse.agent.cortex.transitions import emit_transition_event


def test_ask_question_step_falls_back_to_response_key() -> None:
    state = {
        "locale": "es-MX",
        "channel_type": "telegram",
        "chat_id": "123",
        "last_user_message": "recuérdame en 1 min",
    }
    step = {"tool": "askQuestion", "parameters": {"slot": "time_text"}}
    loop_state = {"kind": "discovery_loop", "steps": [step]}

    result = execution_helpers.run_ask_question_step(
        state,
        step,
        loop_state,
        0,
        build_pending_interaction=build_pending_interaction,
        pending_interaction_type_slot_fill=PendingInteractionType.SLOT_FILL,
        serialize_pending_interaction=serialize_pending_interaction,
        emit_transition_event=emit_transition_event,
    )

    assert "plans" not in result
    assert result.get("response_text") is None
    assert result.get("response_key") == "clarify.repeat_input"
    pending = result.get("pending_interaction")
    assert isinstance(pending, dict)
    assert pending.get("key") == "time_text"
    assert step.get("status") == "waiting"
