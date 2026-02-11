from __future__ import annotations

from alphonse.agent.cognition.capability_gaps.reporting import build_daily_report


def test_daily_report_rendering_en() -> None:
    gaps = [
        {"reason": "unknown_intent", "status": "open", "user_text": "Sync my calendar"},
        {"reason": "missing_slots", "status": "open", "user_text": "Schedule"},
    ]
    report = build_daily_report("en-US", gaps)
    assert "report.daily_gaps.header" in report
    assert "report.daily_gaps.line" in report


def test_daily_report_rendering_es() -> None:
    gaps = [
        {"reason": "no_tool", "status": "open", "user_text": "Conecta calendario"},
    ]
    report = build_daily_report("es-MX", gaps)
    assert "report.daily_gaps.header" in report
    assert "report.daily_gaps.line" in report
