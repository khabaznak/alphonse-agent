"""Candidate events cannot use the unfenced production delivery sink."""

import pytest

from alphonse.agent_v2.core.io.outbox import build_outbox_delivery_sink


class NoSideEffects:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected side effect: {name}")


@pytest.mark.parametrize("event_type", [
    "task.acknowledge", "question.deliver", "communication.resolve_recipient",
    "communication.deliver", "communication.delivery_status", "communication.delivery_timeout",
])
@pytest.mark.parametrize("task", [
    {"intelligence_engine": "single_loop_experimental"},
    {"metadata": {"intelligence_engine": "single_loop_experimental"}},
])
def test_candidate_events_refused_before_delivery_side_effects(event_type, task):
    sink = build_outbox_delivery_sink(
        outbox=NoSideEffects(), identity_resolver=NoSideEffects(),
        communication_router=NoSideEffects(), conversation_store=NoSideEffects(),
    )
    with pytest.raises(ValueError, match="candidate_delivery_unavailable"):
        sink({"event_type": event_type, "task": task})


def test_legacy_unrecognized_event_is_still_ignored():
    sink = build_outbox_delivery_sink(outbox=NoSideEffects(), identity_resolver=NoSideEffects())
    assert sink({"event_type": "unknown", "task": {"intelligence_engine": "hierarchical_v3"}}) == {"ignored": True}
