from __future__ import annotations

from typing import Any


def render_response_key(
    key: str | None,
    response_vars: dict[str, Any] | None = None,
    *,
    locale: str | None = None,
) -> str:
    normalized_key = str(key or "").strip()
    if not normalized_key:
        return ""
    vars = response_vars if isinstance(response_vars, dict) else {}
    lang = "es" if str(locale or "").lower().startswith("es") else "en"
    if normalized_key == "core.greeting":
        return "Hola. ¿En qué te ayudo?" if lang == "es" else "Hello. How can I help?"
    if normalized_key == "clarify.repeat_input":
        return (
            "No te entendí. ¿Podrías repetirlo?"
            if lang == "es"
            else "I did not catch that. Could you repeat it?"
        )
    if normalized_key == "clarify.intent":
        return (
            "¿Qué te gustaría que haga?"
            if lang == "es"
            else "What would you like me to do?"
        )
    if normalized_key == "help":
        return (
            "Puedo ayudarte con recordatorios, estado y ubicación."
            if lang == "es"
            else "I can help with reminders, status, and location."
        )
    if normalized_key == "status":
        return (
            "Estoy en línea y listo para ayudarte."
            if lang == "es"
            else "I am online and ready to help."
        )
    if normalized_key == "ack.cancelled":
        return "Hecho, cancelado." if lang == "es" else "Done, cancelled."
    if normalized_key == "core.location.current.ask_label":
        return (
            "¿Qué ubicación quieres consultar? (home, work u other)"
            if lang == "es"
            else "Which location should I check? (home, work, or other)"
        )
    if normalized_key == "core.location.current.not_set":
        label = str(vars.get("label") or "").strip()
        if label:
            return (
                f"No tengo guardada tu ubicación de {label}."
                if lang == "es"
                else f"I do not have your {label} location saved."
            )
        return (
            "No tengo esa ubicación guardada."
            if lang == "es"
            else "I do not have that location saved."
        )
    if normalized_key == "core.location.current":
        location = str(vars.get("location") or "").strip()
        if location:
            return (
                f"Tu ubicación actual es: {location}."
                if lang == "es"
                else f"Your current location is: {location}."
            )
    if normalized_key == "core.location.set.ask_label":
        return (
            "¿Cuál etiqueta de ubicación quieres guardar? (home, work u other)"
            if lang == "es"
            else "Which location label should I save? (home, work, or other)"
        )
    if normalized_key == "core.location.set.ask_address":
        return (
            "¿Cuál es la dirección que quieres guardar?"
            if lang == "es"
            else "What address should I save?"
        )
    if normalized_key in {"core.location.set.completed", "ack.location.saved"}:
        label = str(vars.get("label") or "").strip()
        if label:
            return (
                f"Listo, guardé la ubicación '{label}'."
                if lang == "es"
                else f"Done, I saved the '{label}' location."
            )
        return (
            "Listo, guardé la ubicación."
            if lang == "es"
            else "Done, I saved the location."
        )
    if normalized_key == "core.users.list":
        lines = vars.get("lines")
        if isinstance(lines, list):
            rendered = "\n".join(str(item) for item in lines if str(item).strip())
            if rendered.strip():
                return rendered
    if normalized_key == "ack.preference_updated":
        return (
            "Listo, actualicé tus preferencias."
            if lang == "es"
            else "Done, I updated your preferences."
        )
    return (
        "Lo siento, no pude resolver esa respuesta todavía."
        if lang == "es"
        else "Sorry, I could not resolve that response yet."
    )
