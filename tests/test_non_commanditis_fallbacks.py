from __future__ import annotations

from alphonse.agent.cognition.localization import render_message


def test_generic_unknown_is_neutral_en() -> None:
    message = render_message("generic.unknown", "en-US")
    lowered = message.lower()
    assert "remind" not in lowered
    assert "reminder" not in lowered


def test_clarify_intent_is_neutral_en() -> None:
    message = render_message("clarify.intent", "en-US")
    lowered = message.lower()
    assert "remind" not in lowered
    assert "reminder" not in lowered


def test_generic_unknown_is_neutral_es() -> None:
    message = render_message("generic.unknown", "es-MX")
    lowered = message.lower()
    assert "recordatorio" not in lowered
    assert "recordatorios" not in lowered


def test_clarify_intent_is_neutral_es() -> None:
    message = render_message("clarify.intent", "es-MX")
    lowered = message.lower()
    assert "recordatorio" not in lowered
    assert "recordatorios" not in lowered
