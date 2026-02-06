from __future__ import annotations

from typing import Any


def render_message(
    key: str, locale: str, variables: dict[str, Any] | None = None
) -> str:
    vars = variables or {}
    language = _language_for(locale)
    if key == "greeting":
        return _render_greeting(language, vars)
    if key == "ack.preference_updated":
        return _render_preference_ack(language, vars)
    if key == "greeting":
        return _render_greeting(language, vars)
    if key == "clarify.reminder_text":
        return _render_clarify_reminder_text(language, vars)
    if key == "clarify.trigger_time":
        return _render_clarify_trigger_time(language, vars)
    template = _template_for_key(language, key, vars)
    if template is None:
        template = _TEMPLATES["en"].get("generic.unknown")
    assert template is not None
    return template.format(**vars)


def _language_for(locale: str) -> str:
    return "es" if locale and str(locale).lower().startswith("es") else "en"


def _render_preference_ack(language: str, vars: dict[str, Any]) -> str:
    updates = vars.get("updates") or []
    address_style = _normalize_address(vars.get("address_style"))
    parts: list[str] = []
    for update in updates:
        key = update.get("key")
        value = update.get("value")
        if key == "address_style":
            address_style = _normalize_address(value)
            if language == "es":
                parts.append(
                    "Listo, le hablaré de usted."
                    if address_style == "usted"
                    else "Listo, te hablaré de tú."
                )
            else:
                parts.append(
                    "Got it. I'll address you formally."
                    if address_style == "usted"
                    else "Got it. I'll keep it casual."
                )
        elif key == "locale":
            if language == "es":
                parts.append(
                    "Listo, hablaré en español."
                    if str(value).startswith("es")
                    else "Listo, hablaré en inglés."
                )
            else:
                parts.append(
                    "Got it. I'll use Spanish."
                    if str(value).startswith("es")
                    else "Got it. I'll use English."
                )
        elif key == "tone":
            if language == "es":
                parts.append(
                    "Seré más formal." if value == "formal" else "Seré más casual."
                )
            else:
                parts.append(
                    "Got it. I'll be more formal."
                    if value == "formal"
                    else "Got it. I'll be more casual."
                )
    if parts:
        return " ".join(parts)
    if language == "es":
        return "Listo, ajusté tus preferencias."
    return "Got it. Preferences updated."


def _render_greeting(language: str, vars: dict[str, Any]) -> str:
    address_style = _normalize_address(vars.get("address_style"))
    if language == "es":
        if address_style == "usted":
            return "Hola. ¿En qué puedo ayudarle?"
        return "¡Hola! ¿En qué te ayudo?"
    if _is_formal(vars.get("tone")):
        return "Hello. How can I help?"
    return "Hi. How can I help?"


def _template_for_key(language: str, key: str, vars: dict[str, Any]) -> str | None:
    if language == "es" and _normalize_address(vars.get("address_style")) == "usted":
        template = _FORMAL_ES_TEMPLATES.get(key)
        if template is not None:
            return template
    return _TEMPLATES.get(language, {}).get(key)


def _render_clarify_reminder_text(language: str, vars: dict[str, Any]) -> str:
    address_style = _normalize_address(vars.get("address_style"))
    if language == "es":
        if address_style == "usted":
            return '¿Qué debo recordarle? Por ejemplo: "llamar a mamá".'
        return '¿Qué debo recordarte? Por ejemplo: "llamar a mamá".'
    if _is_formal(vars.get("tone")):
        return 'What should I remind you about? For example: "call mom".'
    return 'What should I remind you about? For example: "call mom".'


def _render_clarify_trigger_time(language: str, vars: dict[str, Any]) -> str:
    address_style = _normalize_address(vars.get("address_style"))
    if language == "es":
        if address_style == "usted":
            return '¿Cuándo debo recordárselo? Ejemplo: "en 10 min" o "a las 7pm".'
        return '¿Cuándo debo recordarlo? Ejemplo: "en 10 min" o "a las 7pm".'
    if _is_formal(vars.get("tone")):
        return 'When should I remind you? Example: "in 10 min" or "at 7pm".'
    return 'When should I remind you? Example: "in 10 min" or "at 7pm".'


def _normalize_address(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in {"tu", "usted"}:
        return value.strip().lower()
    return "tu"


def _is_formal(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "formal"


_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "generic.unknown": 'I can help with reminders. Try: "Remind me to drink water in 10 min".',
        "clarify.intent": "I'm not sure what you need yet. What would you like me to do?",
        "help": 'I can schedule reminders. Try: "Remind me to drink water in 10 min".',
        "status": "I'm active and ready for reminders.",
        "identity": "I'm Alphonse, your assistant. I only know this authorized chat.",
        "identity.user": "I don't know your name yet. Tell me what you'd like me to call you.",
        "ack.user_name": "Perfect, {user_name}. I'll call you that from now on.",
        "ack.confirmed": "Got it.",
        "ack.reminder_scheduled": "Got it. Reminder scheduled.",
        "preference.missing": "What preference should I update?",
        "preference.no_channel": "I need a channel to store your preferences.",
        "report.daily_gaps.header": "Daily gap report: {total} total, {open} open.",
        "report.daily_gaps.line": "- {reason} ({count}): {example}",
        "report.daily_gaps.empty": "No capability gaps in the last day.",
    },
    "es": {
        "generic.unknown": 'Puedo programar recordatorios. Prueba: "Recuérdame tomar agua en 10 min".',
        "clarify.intent": "No estoy seguro de lo que necesitas. ¿Qué te gustaría que hiciera?",
        "help": 'Puedo programar recordatorios. Ejemplo: "Recuérdame tomar agua en 10 min".',
        "status": "Estoy activo y listo para recordatorios.",
        "identity": "Soy Alphonse, tu asistente. Solo conozco este chat autorizado.",
        "identity.user": "Aún no sé tu nombre. Dime cómo quieres que te llame.",
        "ack.user_name": "Perfecto, {user_name}. A partir de ahora te llamaré así.",
        "ack.confirmed": "Entendido.",
        "ack.reminder_scheduled": "Listo, programé el recordatorio.",
        "preference.missing": "¿Qué preferencia quieres ajustar?",
        "preference.no_channel": "Necesito un canal para guardar tus preferencias.",
        "report.daily_gaps.header": "Reporte diario de brechas: {total} total, {open} abiertas.",
        "report.daily_gaps.line": "- {reason} ({count}): {example}",
        "report.daily_gaps.empty": "No hubo brechas de capacidad en el último día.",
    },
}

_FORMAL_ES_TEMPLATES: dict[str, str] = {
    "generic.unknown": 'Puedo programar recordatorios. Pruebe: "Recuérdeme tomar agua en 10 min".',
    "clarify.intent": "No estoy seguro de lo que necesita. ¿Qué le gustaría que hiciera?",
    "help": 'Puedo programar recordatorios. Ejemplo: "Recuérdeme tomar agua en 10 min".',
    "status": "Estoy activo y listo para recordatorios.",
    "identity": "Soy Alphonse, su asistente. Solo conozco este chat autorizado.",
    "identity.user": "Aún no sé su nombre. Dígame cómo quiere que le llame.",
    "ack.user_name": "Perfecto, {user_name}. A partir de ahora le llamaré así.",
    "ack.confirmed": "Entendido.",
    "ack.reminder_scheduled": "Listo, programé el recordatorio.",
    "preference.missing": "¿Qué preferencia quiere ajustar?",
    "preference.no_channel": "Necesito un canal para guardar sus preferencias.",
    "report.daily_gaps.header": "Reporte diario de brechas: {total} total, {open} abiertas.",
    "report.daily_gaps.line": "- {reason} ({count}): {example}",
    "report.daily_gaps.empty": "No hubo brechas de capacidad en el último día.",
}
