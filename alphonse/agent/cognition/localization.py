from __future__ import annotations

from typing import Any


def render_message(
    key: str, locale: str, variables: dict[str, Any] | None = None
) -> str:
    vars = variables or {}
    language = _language_for(locale)
    if key in {"greeting", "core.greeting"}:
        return _render_greeting(language, vars)
    if key == "ack.preference_updated":
        return _render_preference_ack(language, vars)
    if key in {"greeting", "core.greeting"}:
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
        "generic.unknown": "I'm not sure what you mean yet. What would you like to do?",
        "clarify.intent": "I'm not sure what you mean yet. What would you like to do?",
        "clarify.slot_abort": "We can cancel this or try again later. What would you prefer?",
        "help": 'I can schedule reminders. Try: "Remind me to drink water in 10 min".',
        "status": "I'm active and ready.",
        "identity": "I'm Alphonse, your assistant. I only know this authorized chat.",
        "identity.user": "I don't know your name yet. Tell me what you'd like me to call you.",
        "identity.user.known": "Yes, your name is {user_name}.",
        "core.identity.agent": "I'm Alphonse, your assistant. I only know this authorized chat.",
        "core.identity.user.ask_name": "I don't know your name yet. Tell me what you'd like me to call you.",
        "core.identity.user.known": "Yes, your name is {user_name}.",
        "ack.user_name": "Perfect, {user_name}. I'll call you that from now on.",
        "ack.confirmed": "Got it.",
        "ack.reminder_scheduled": "Got it. Reminder scheduled.",
        "ack.cancelled": "Okay, I cancelled that.",
        "clarify.trigger_geo.stub_setup": "I can do location-based reminders once home is set up. Want to switch to a time-based reminder?",
        "preference.missing": "What preference should I update?",
        "preference.no_channel": "I need a channel to store your preferences.",
        "lan.device.not_found": "No paired devices found.",
        "lan.device.armed": "✅ Armed device {device_name} ({device_id})",
        "lan.device.disarmed": "🔒 Disarmed device {device_name} ({device_id})",
        "pairing.not_found": "Pairing not found.",
        "pairing.already_resolved": "Pairing already {status}.",
        "pairing.missing_otp": "Missing OTP.",
        "pairing.approved": "✅ Pairing approved.",
        "pairing.invalid_otp": "Invalid OTP or expired.",
        "pairing.denied": "Pairing denied.",
        "error.execution_failed": "Sorry, I had a problem processing that request.",
        "policy.reminder.restricted": "I am not authorized to schedule that reminder.",
        "report.daily_gaps.header": "Daily gap report: {total} total, {open} open.",
        "report.daily_gaps.line": "- {reason} ({count}): {example}",
        "report.daily_gaps.empty": "No capability gaps in the last day.",
    },
    "es": {
        "generic.unknown": "No estoy seguro de a qué te refieres. ¿Qué te gustaría hacer?",
        "clarify.intent": "No estoy seguro de a qué te refieres. ¿Qué te gustaría hacer?",
        "clarify.slot_abort": "Puedo cancelar esto o intentarlo más tarde. ¿Qué prefieres?",
        "help": 'Puedo programar recordatorios. Ejemplo: "Recuérdame tomar agua en 10 min".',
        "status": "Estoy activo y listo.",
        "identity": "Soy Alphonse, tu asistente. Solo conozco este chat autorizado.",
        "identity.user": "Aún no sé tu nombre. Dime cómo quieres que te llame.",
        "identity.user.known": "Sí, te llamas {user_name}.",
        "core.identity.agent": "Soy Alphonse, tu asistente. Solo conozco este chat autorizado.",
        "core.identity.user.ask_name": "Aún no sé tu nombre. Dime cómo quieres que te llame.",
        "core.identity.user.known": "Sí, te llamas {user_name}.",
        "ack.user_name": "Perfecto, {user_name}. A partir de ahora te llamaré así.",
        "ack.confirmed": "Entendido.",
        "ack.reminder_scheduled": "Listo, programé el recordatorio.",
        "ack.cancelled": "Listo, lo cancelé.",
        "clarify.trigger_geo.stub_setup": "Puedo hacer recordatorios por ubicación cuando casa esté configurada. ¿Quieres usar un recordatorio por hora?",
        "preference.missing": "¿Qué preferencia quieres ajustar?",
        "preference.no_channel": "Necesito un canal para guardar tus preferencias.",
        "lan.device.not_found": "No se encontraron dispositivos emparejados.",
        "lan.device.armed": "✅ Dispositivo armado {device_name} ({device_id})",
        "lan.device.disarmed": "🔒 Dispositivo desarmado {device_name} ({device_id})",
        "pairing.not_found": "No encontré ese emparejamiento.",
        "pairing.already_resolved": "El emparejamiento ya está {status}.",
        "pairing.missing_otp": "Falta el OTP.",
        "pairing.approved": "✅ Emparejamiento aprobado.",
        "pairing.invalid_otp": "OTP inválido o expirado.",
        "pairing.denied": "Emparejamiento rechazado.",
        "error.execution_failed": "Lo siento, tuve un problema al procesar la solicitud.",
        "policy.reminder.restricted": "No estoy autorizado para programar ese recordatorio.",
        "report.daily_gaps.header": "Reporte diario de brechas: {total} total, {open} abiertas.",
        "report.daily_gaps.line": "- {reason} ({count}): {example}",
        "report.daily_gaps.empty": "No hubo brechas de capacidad en el último día.",
    },
}

_FORMAL_ES_TEMPLATES: dict[str, str] = {
    "generic.unknown": "No estoy seguro de a qué se refiere. ¿Qué le gustaría hacer?",
    "clarify.intent": "No estoy seguro de a qué se refiere. ¿Qué le gustaría hacer?",
    "clarify.slot_abort": "Puedo cancelar esto o intentarlo más tarde. ¿Qué prefiere?",
    "help": 'Puedo programar recordatorios. Ejemplo: "Recuérdeme tomar agua en 10 min".',
    "status": "Estoy activo y listo.",
    "identity": "Soy Alphonse, su asistente. Solo conozco este chat autorizado.",
    "identity.user": "Aún no sé su nombre. Dígame cómo quiere que le llame.",
    "identity.user.known": "Sí, se llama {user_name}.",
    "core.identity.agent": "Soy Alphonse, su asistente. Solo conozco este chat autorizado.",
    "core.identity.user.ask_name": "Aún no sé su nombre. Dígame cómo quiere que le llame.",
    "core.identity.user.known": "Sí, se llama {user_name}.",
    "ack.user_name": "Perfecto, {user_name}. A partir de ahora le llamaré así.",
    "ack.confirmed": "Entendido.",
    "ack.reminder_scheduled": "Listo, programé el recordatorio.",
    "ack.cancelled": "Listo, lo cancelé.",
    "clarify.trigger_geo.stub_setup": "Puedo hacer recordatorios por ubicación cuando casa esté configurada. ¿Quiere usar un recordatorio por hora?",
    "preference.missing": "¿Qué preferencia quiere ajustar?",
    "preference.no_channel": "Necesito un canal para guardar sus preferencias.",
    "lan.device.not_found": "No se encontraron dispositivos emparejados.",
    "lan.device.armed": "✅ Dispositivo armado {device_name} ({device_id})",
    "lan.device.disarmed": "🔒 Dispositivo desarmado {device_name} ({device_id})",
    "pairing.not_found": "No encontré ese emparejamiento.",
    "pairing.already_resolved": "El emparejamiento ya está {status}.",
    "pairing.missing_otp": "Falta el OTP.",
    "pairing.approved": "✅ Emparejamiento aprobado.",
    "pairing.invalid_otp": "OTP inválido o expirado.",
    "pairing.denied": "Emparejamiento rechazado.",
    "error.execution_failed": "Lo siento, tuve un problema al procesar la solicitud.",
    "policy.reminder.restricted": "No estoy autorizado para programar ese recordatorio.",
    "report.daily_gaps.header": "Reporte diario de brechas: {total} total, {open} abiertas.",
    "report.daily_gaps.line": "- {reason} ({count}): {example}",
    "report.daily_gaps.empty": "No hubo brechas de capacidad en el último día.",
}
