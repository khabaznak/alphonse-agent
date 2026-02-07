from __future__ import annotations


_SAFE_FALLBACKS: dict[str, dict[str, str]] = {
    "en": {
        "ack.user_name": "Perfect, {user_name}. I'll call you that from now on.",
        "ack.confirmed": "Got it.",
        "generic.unknown": "I'm not sure what you mean yet. What would you like to do?",
        "clarify.intent": "I'm not sure what you mean yet. What would you like to do?",
        "clarify.repeat_input": "I did not catch that. Could you repeat?",
        "ack.preference_updated": "Got it. Preferences updated.",
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
        "clarify.repeat_input": "No te escuche bien. Puedes repetir?",
        "ack.preference_updated": "Listo, ajusté tus preferencias.",
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


def render_safe_message(
    key: str, locale: str | None = None, variables: dict[str, object] | None = None
) -> str:
    language = "es" if str(locale or "").lower().startswith("es") else "en"
    if key == "ack.preference_updated":
        rendered = _render_preference_ack(language, variables or {})
        if rendered:
            return rendered
    template = get_safe_fallback(key, locale)
    try:
        return template.format(**(variables or {}))
    except Exception:
        return template


def _render_preference_ack(language: str, variables: dict[str, object]) -> str:
    updates = variables.get("updates")
    if not isinstance(updates, list) or not updates:
        return get_safe_fallback("ack.preference_updated", "es" if language == "es" else "en")
    parts: list[str] = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        key = str(update.get("key") or "")
        value = str(update.get("value") or "")
        if key == "locale":
            if language == "es":
                parts.append(
                    "Listo, hablaré en español."
                    if value.startswith("es")
                    else "Listo, hablaré en inglés."
                )
            else:
                parts.append(
                    "Got it. I'll use Spanish."
                    if value.startswith("es")
                    else "Got it. I'll use English."
                )
        elif key == "address_style":
            if language == "es":
                parts.append(
                    "Listo, le hablaré de usted."
                    if value == "usted"
                    else "Listo, te hablaré de tú."
                )
            else:
                parts.append(
                    "Got it. I'll address you formally."
                    if value == "usted"
                    else "Got it. I'll keep it casual."
                )
        elif key == "tone":
            if language == "es":
                parts.append("Seré más formal." if value == "formal" else "Seré más casual.")
            else:
                parts.append(
                    "Got it. I'll be more formal."
                    if value == "formal"
                    else "Got it. I'll be more casual."
                )
    if parts:
        return " ".join(parts)
    return get_safe_fallback("ack.preference_updated", "es" if language == "es" else "en")
