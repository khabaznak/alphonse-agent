from __future__ import annotations

from alphonse.agent.cognition.capability_gaps.reporting import build_daily_report


def test_daily_report_rendering_en() -> None:
    gaps = [
        {"reason": "unknown_intent", "status": "open", "user_text": "Sync my calendar"},
        {"reason": "missing_slots", "status": "open", "user_text": "Schedule"},
    ]
    report = build_daily_report("en-US", gaps)
    assert "Daily capability gaps: total=2, open=2" in report
    assert "reason=unknown intent" in report
    assert "reason=missing slots" in report
    assert "Sync my calendar" in report


def test_daily_report_rendering_es() -> None:
    gaps = [
        {"reason": "no_tool", "status": "open", "user_text": "Conecta calendario"},
    ]
    report = build_daily_report("es-MX", gaps)
    assert "Daily capability gaps: total=1, open=1" in report
    assert "reason=no tool" in report
    assert "Conecta calendario" in report
