from __future__ import annotations

from datetime import datetime, timezone

from alphonse.agent.cognition.slots.resolvers import TimeExpressionResolver


def test_time_expression_parses_english_minutes() -> None:
    resolver = TimeExpressionResolver()
    now = datetime(2026, 2, 6, 12, 0, tzinfo=timezone.utc)
    result = resolver.parse("fifteen minutes", {"timezone": "UTC", "now": now})
    assert result.ok is True
    assert result.value["kind"] == "trigger_at"


def test_time_expression_parses_spanish_minutes() -> None:
    resolver = TimeExpressionResolver()
    now = datetime(2026, 2, 6, 12, 0, tzinfo=timezone.utc)
    result = resolver.parse("quince minutos", {"timezone": "UTC", "now": now})
    assert result.ok is True
    assert result.value["kind"] == "trigger_at"


def test_time_expression_parses_numeric() -> None:
    resolver = TimeExpressionResolver()
    now = datetime(2026, 2, 6, 12, 0, tzinfo=timezone.utc)
    result = resolver.parse("15min", {"timezone": "UTC", "now": now})
    assert result.ok is True
    assert result.value["kind"] == "trigger_at"


def test_time_expression_invalid() -> None:
    resolver = TimeExpressionResolver()
    now = datetime(2026, 2, 6, 12, 0, tzinfo=timezone.utc)
    result = resolver.parse("sometime laterish", {"timezone": "UTC", "now": now})
    assert result.ok is False
