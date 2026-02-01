from __future__ import annotations

from alphonse.agent.cognition.capability_gaps.reporting import build_daily_report


def test_daily_report_rendering_en() -> None:
    gaps = [
        {"reason": "unknown_intent", "status": "open", "user_text": "Sync my calendar"},
        {"reason": "missing_slots", "status": "open", "user_text": "Schedule"},
    ]
    report = build_daily_report("en-US", gaps)
    assert "Daily gap report" in report
    assert "unknown intent" in report


def test_daily_report_rendering_es() -> None:
    gaps = [
        {"reason": "no_tool", "status": "open", "user_text": "Conecta calendario"},
    ]
    report = build_daily_report("es-MX", gaps)
    assert "Reporte diario" in report
    assert "sin herramienta" in report
