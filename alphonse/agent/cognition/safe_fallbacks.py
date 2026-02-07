from __future__ import annotations


_SAFE_FALLBACKS: dict[str, dict[str, str]] = {
    "en": {
        "ack.user_name": "Perfect, {user_name}. I'll call you that from now on.",
        "ack.confirmed": "Got it.",
        "generic.unknown": "I'm not sure what you mean yet. What would you like to do?",
        "clarify.intent": "I'm not sure what you mean yet. What would you like to do?",
        "report.daily_gaps.header": "Daily gap report: {total} total, {open} open.",
        "report.daily_gaps.line": "- {reason} ({count}): {example}",
        "report.daily_gaps.empty": "No capability gaps in the last day.",
        "policy.reminder.restricted": "I am not authorized to schedule that reminder.",
        "error.execution_failed": "Sorry, I had a problem processing that request.",
        "system.unavailable.catalog": "Alphonse is unavailable: intent catalog is not ready.",
        "system.unavailable.prompt_store": "Alphonse is unavailable: prompt store is not ready.",
        "system.unavailable.nerve_db": "Alphonse is unavailable: nerve-db is not accessible.",
        "default": "I can't do that right now.",
    },
    "es": {
        "ack.user_name": "Perfecto, {user_name}. A partir de ahora te llamaré así.",
        "ack.confirmed": "Entendido.",
        "generic.unknown": "No estoy seguro de a qué te refieres. ¿Qué te gustaría hacer?",
        "clarify.intent": "No estoy seguro de a qué te refieres. ¿Qué te gustaría hacer?",
        "report.daily_gaps.header": "Reporte diario de brechas: {total} total, {open} abiertas.",
        "report.daily_gaps.line": "- {reason} ({count}): {example}",
        "report.daily_gaps.empty": "No hubo brechas de capacidad en el último día.",
        "policy.reminder.restricted": "No estoy autorizado para programar ese recordatorio.",
        "error.execution_failed": "Lo siento, tuve un problema al procesar la solicitud.",
        "system.unavailable.catalog": "Alphonse no está disponible: el catálogo de intents no está listo.",
        "system.unavailable.prompt_store": "Alphonse no está disponible: el almacén de prompts no está listo.",
        "system.unavailable.nerve_db": "Alphonse no está disponible: no hay acceso a nerve-db.",
        "default": "No puedo hacer eso ahora mismo.",
    },
}


def get_safe_fallback(key: str, locale: str | None = None) -> str:
    language = "es" if str(locale or "").lower().startswith("es") else "en"
    templates = _SAFE_FALLBACKS.get(language, _SAFE_FALLBACKS["en"])
    return templates.get(key) or templates["default"]
