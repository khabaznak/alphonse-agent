from __future__ import annotations


_SAFE_FALLBACKS: dict[str, dict[str, str]] = {
    "en": {
        "policy.reminder.restricted": "I am not authorized to schedule that reminder.",
        "error.execution_failed": "Sorry, I had a problem processing that request.",
        "system.unavailable.catalog": "Alphonse is unavailable: intent catalog is not ready.",
        "system.unavailable.prompt_store": "Alphonse is unavailable: prompt store is not ready.",
        "system.unavailable.nerve_db": "Alphonse is unavailable: nerve-db is not accessible.",
        "default": "I can't do that right now.",
    },
    "es": {
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

