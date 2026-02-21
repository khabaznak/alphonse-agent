from __future__ import annotations

from collections import Counter
from typing import Any

from alphonse.agent.nervous_system.capability_gaps import list_gaps
from alphonse.agent.nervous_system.timed_store import list_upcoming_timed_signals


def summarize_gaps(locale: str, limit: int = 5) -> str:
    gaps = list_gaps(status="open", limit=50, include_all=False)
    if not gaps:
        return _text(locale, "gaps.empty")
    reasons = Counter([gap.get("reason") or "unknown" for gap in gaps])
    total = len(gaps)
    top = list_gaps(status="open", limit=limit, include_all=False)
    lines = [_text(locale, "gaps.header").format(total=total)]
    for reason, count in reasons.most_common():
        lines.append(f"- {reason}: {count}")
    if top:
        lines.append(_text(locale, "gaps.recent"))
        for gap in top:
            snippet = _snippet(str(gap.get("user_text") or ""))
            lines.append(f"- {snippet}")
    return "\n".join(lines)


def summarize_capabilities(locale: str) -> str:
    lines = [
        _text(locale, "capabilities.header"),
        _text(locale, "capabilities.items"),
        _text(locale, "capabilities.examples"),
    ]
    return "\n".join(lines)


def summarize_timed_signals(locale: str, limit: int = 10) -> str:
    signals = list_upcoming_timed_signals(limit=limit)
    if not signals:
        return _text(locale, "timed_signals.empty")
    lines = [_text(locale, "timed_signals.header").format(total=len(signals))]
    for signal in signals:
        trigger_at = signal.get("next_trigger_at") or signal.get("trigger_at") or "unknown"
        payload = signal.get("payload") or {}
        message = (
            payload.get("prompt")
            or payload.get("agent_internal_prompt")
            or payload.get("prompt_text")
            or payload.get("message")
            or payload.get("message_text")
            or payload.get("reminder_text")
            or "timed_signal"
        )
        lines.append(f"- {trigger_at}: {_snippet(str(message))}")
    return "\n".join(lines)


def _snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _text(locale: str, key: str) -> str:
    language = "es" if str(locale).lower().startswith("es") else "en"
    return _TEXT.get(language, _TEXT["en"]).get(key, key)


_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "gaps.header": "Open gaps: {total}",
        "gaps.recent": "Most recent:",
        "gaps.empty": "No open gaps right now.",
        "capabilities.header": "Current capabilities:",
        "capabilities.items": "- Schedule reminders\n- List upcoming reminders\n- Show gaps and proposals",
        "capabilities.examples": 'Examples: "Remind me to drink water in 10 min", "What reminders do you have scheduled?", "gaps list"',
        "timed_signals.header": "Upcoming reminders (showing {total}):",
        "timed_signals.empty": "No upcoming reminders.",
    },
    "es": {
        "gaps.header": "Brechas abiertas: {total}",
        "gaps.recent": "Más recientes:",
        "gaps.empty": "No hay brechas abiertas.",
        "capabilities.header": "Capacidades actuales:",
        "capabilities.items": "- Programar recordatorios\n- Listar recordatorios\n- Ver brechas y propuestas",
        "capabilities.examples": 'Ejemplos: "Recuérdame tomar agua en 10 min", "¿Qué recordatorios tienes?", "gaps list"',
        "timed_signals.header": "Próximos recordatorios (mostrando {total}):",
        "timed_signals.empty": "No hay recordatorios próximos.",
    },
}
