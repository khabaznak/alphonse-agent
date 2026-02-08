from __future__ import annotations

from alphonse.agent.cognition.pending_interaction import (
    PendingInteraction,
    PendingInteractionType,
    try_consume,
)


def test_name_slot_fill_consumes() -> None:
    pending = PendingInteraction(
        type=PendingInteractionType.SLOT_FILL,
        key="user_name",
        context={"origin_intent": "identity.learn_user_name"},
        created_at="now",
        expires_at=None,
    )
    result = try_consume("Alex", pending)
    assert result.consumed is True
    assert result.result == {"user_name": "Alex"}


def test_name_slot_fill_rejects_empty() -> None:
    pending = PendingInteraction(
        type=PendingInteractionType.SLOT_FILL,
        key="user_name",
        context={},
        created_at="now",
        expires_at=None,
    )
    result = try_consume("", pending)
    assert result.consumed is False


def test_confirmation_yes_no() -> None:
    pending = PendingInteraction(
        type=PendingInteractionType.CONFIRMATION,
        key="plan_confirmation",
        context={},
        created_at="now",
        expires_at=None,
    )
    yes = try_consume("sí", pending)
    no = try_consume("no", pending)
    assert yes.consumed is True
    assert yes.result == {"confirmed": True}
    assert no.consumed is True
    assert no.result == {"confirmed": False}


def test_address_slot_fill_consumes() -> None:
    pending = PendingInteraction(
        type=PendingInteractionType.SLOT_FILL,
        key="address_text",
        context={"label": "home"},
        created_at="now",
        expires_at=None,
    )
    result = try_consume("123 Main St", pending)
    assert result.consumed is True
    assert result.result == {"address_text": "123 Main St"}


def test_address_slot_fill_rejects_short() -> None:
    pending = PendingInteraction(
        type=PendingInteractionType.SLOT_FILL,
        key="address_text",
        context={},
        created_at="now",
        expires_at=None,
    )
    result = try_consume("Apt", pending)
    assert result.consumed is False
