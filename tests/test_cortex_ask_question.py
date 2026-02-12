from alphonse.agent.cortex import graph


def test_ask_question_step_falls_back_to_response_key() -> None:
    state = {
        "locale": "es-MX",
        "channel_type": "telegram",
        "chat_id": "123",
        "last_user_message": "recuérdame en 1 min",
    }
    step = {"tool": "askQuestion", "parameters": {"slot": "time_text"}}
    loop_state = {"kind": "discovery_loop", "steps": [step]}

    result = graph._run_ask_question_step(state, step, loop_state, 0)

    assert "plans" not in result
    assert result.get("response_text") is None
    assert result.get("response_key") == "clarify.repeat_input"
    pending = result.get("pending_interaction")
    assert isinstance(pending, dict)
    assert pending.get("key") == "time_text"
    assert step.get("status") == "waiting"
