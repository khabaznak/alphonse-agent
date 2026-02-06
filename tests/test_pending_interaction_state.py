from __future__ import annotations

from alphonse.agent.cortex.graph import invoke_cortex


def test_user_identity_sets_pending_interaction() -> None:
    state = {
        "chat_id": "telegram:123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "America/Mexico_City",
    }
    result = invoke_cortex(state, "Sabes como me llamo yo?", llm_client=None)
    assert result.meta.get("response_key") == "identity.user"
    assert result.cognition_state.get("pending_interaction") is not None
