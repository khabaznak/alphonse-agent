from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_with_fallback,
)
from alphonse.config import settings

logger = logging.getLogger(__name__)


def render_reminder(
    payload: dict[str, Any], prefs: dict[str, Any] | None = None
) -> str:
    stored_prefs = _load_preferences(payload)
    prefs = {**stored_prefs, **(prefs or {})}
    raw = _extract_raw_text(payload)
    cleaned = _normalize_task_text(raw)
    locale = _resolve_locale(payload, cleaned, prefs)
    tone = _resolve_tone(prefs)
    address_style = _resolve_address_style(prefs)
    name = _resolve_name(payload, prefs)
    timing = _timing_reference(payload, locale)
    relay_style = prefs.get("reminders.relay_style", True)
    if relay_style is False:
        rendered = _render_simple(cleaned, locale)
    else:
        rendered = _render_relay(
            cleaned,
            locale=locale,
            tone=tone,
            address_style=address_style,
            name=name,
            timing=timing,
        )
    logger.info(
        "ReminderRenderer locale=%s tone=%s address=%s raw_len=%s cleaned_len=%s",
        locale,
        tone,
        address_style,
        len(raw or ""),
        len(cleaned),
    )
    return rendered


def _extract_raw_text(payload: dict[str, Any]) -> str:
    if isinstance(payload, dict):
        raw = payload.get("reminder_text_raw") or payload.get("message")
        if raw:
            return str(raw).strip()
    return ""


def _normalize_task_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "esto"
    cleaned = _strip_leading_prompt(cleaned)
    cleaned = cleaned.strip()
    if cleaned and cleaned[0].isupper() and (len(cleaned) == 1 or cleaned[1].islower()):
        cleaned = cleaned[0].lower() + cleaned[1:]
    cleaned = _safe_light_corrections(cleaned)
    cleaned = _strip_terminal_punctuation(cleaned)
    cleaned = _normalize_spanish_verb(cleaned)
    cleaned = _normalize_english_verb(cleaned)
    return cleaned or "esto"


def _safe_light_corrections(text: str) -> str:
    replacements = {
        r"\brecuarda\b": "recuerda",
        r"\brecuerda\b": "recuerda",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _resolve_locale(payload: dict[str, Any], text: str, prefs: dict[str, Any]) -> str:
    hint = payload.get("locale_hint") if isinstance(payload, dict) else None
    if isinstance(hint, str) and hint.strip():
        return hint
    pref_locale = prefs.get("locale") if isinstance(prefs, dict) else None
    if isinstance(pref_locale, str) and pref_locale.strip():
        return pref_locale
    lowered = text.lower()
    if any(
        token in lowered
        for token in ("recuérd", "recuerda", "mañana", "hoy", "bañar", "bañ")
    ):
        return "es-MX"
    if any(
        token in lowered
        for token in ("remind", "tomorrow", "today", "please", "lunch", "family")
    ):
        return "en-US"
    return settings.get_default_locale()


def _resolve_tone(prefs: dict[str, Any]) -> str:
    pref_tone = prefs.get("tone") if isinstance(prefs, dict) else None
    if isinstance(pref_tone, str) and pref_tone.strip():
        return pref_tone.strip()
    return settings.get_tone()


def _resolve_address_style(prefs: dict[str, Any]) -> str:
    pref_style = prefs.get("address_style") if isinstance(prefs, dict) else None
    if isinstance(pref_style, str) and pref_style.strip():
        style = pref_style.strip().lower()
    else:
        style = settings.get_address_style()
    return style if style in {"tu", "usted"} else "tu"


def _resolve_name(payload: dict[str, Any], prefs: dict[str, Any]) -> str | None:
    if isinstance(prefs, dict):
        name = prefs.get("name") or prefs.get("user_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    if isinstance(payload, dict):
        for key in ("user_name", "person_name", "name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _render_relay(
    task: str,
    *,
    locale: str,
    tone: str,
    address_style: str,
    name: str | None,
    timing: str | None,
) -> str:
    _ = tone
    task = task or "esto"
    if locale.startswith("es"):
        name_part = f"{name}, " if name else ""
        if address_style == "usted":
            template = "🕒 {name}me pediste que le recuerde {task}"
        else:
            template = "🕒 {name}me pediste que te recuerde {task}"
        body = template.format(name=name_part, task=task)
    else:
        body = f"🕒 You asked me to remind you to {task}"
    if timing:
        body = f"{body} {timing}"
    return _ensure_terminal_punctuation(body)


def _render_simple(task: str, locale: str) -> str:
    body = task or "recordatorio"
    prefix = "🕒 Recordatorio: " if locale.startswith("es") else "🕒 Reminder: "
    return _ensure_terminal_punctuation(f"{prefix}{body}")


def _ensure_terminal_punctuation(text: str) -> str:
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    return f"{text}."


def _strip_leading_prompt(text: str) -> str:
    lowered = text.lower()
    patterns = [
        r"^(por favor\s+)?(recu[eé]rdame|recuerda|recordarme)\s+(que\s+)?",
        r"^(por favor\s+)?(recuerde)\s+(que\s+)?",
        r"^(please\s+)?(remind\s+me\s+to|remind\s+me\s+that)\s+",
        r"^(please\s+)?(remember\s+to)\s+",
    ]
    for pattern in patterns:
        match = re.match(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return text[match.end() :]
    return text


def _strip_terminal_punctuation(text: str) -> str:
    return text.rstrip(" .!?")


def _normalize_spanish_verb(text: str) -> str:
    if not text:
        return text
    parts = text.split(" ", 1)
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    lowered = first.lower()
    if lowered.endswith(("ar", "er", "ir")):
        return text
    imperative_map = {
        "ve": "ir",
        "haz": "hacer",
        "pon": "poner",
        "di": "decir",
        "ven": "venir",
        "sal": "salir",
        "ten": "tener",
    }
    if lowered in imperative_map:
        replacement = imperative_map[lowered]
        return f"{replacement} {rest}".strip()
    return text


def _normalize_english_verb(text: str) -> str:
    lowered = text.lower()
    if lowered.startswith("to "):
        return text[3:]
    match = re.match(r"^go\s+(\w+)(\b.*)", text, flags=re.IGNORECASE)
    if not match:
        return text
    verb = match.group(1).lower()
    allowed = {
        "prepare",
        "make",
        "buy",
        "take",
        "do",
        "get",
        "fix",
        "cook",
        "clean",
        "call",
        "check",
        "send",
        "write",
        "pay",
        "pick",
        "drop",
        "wash",
    }
    if verb in allowed:
        return f"{match.group(1)}{match.group(2)}".strip()
    return text


def _timing_reference(payload: dict[str, Any], locale: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    trigger_at = payload.get("trigger_at") or payload.get("scheduled_for")
    if not isinstance(trigger_at, str) or not trigger_at.strip():
        return None
    trigger_dt = _parse_datetime(trigger_at.strip())
    if not trigger_dt:
        return None
    now = datetime.now(timezone.utc)
    delta_seconds = abs((now - trigger_dt.astimezone(timezone.utc)).total_seconds())
    if delta_seconds > 120:
        return None
    return "ahora" if locale.startswith("es") else "now"


def _parse_datetime(value: str) -> datetime | None:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        tz_name = settings.get_timezone()
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(tz_name))
        except Exception:
            parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _load_preferences(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    channel_type = payload.get("origin_channel") or payload.get("channel_type")
    channel_id = (
        payload.get("chat_id") or payload.get("channel_id") or payload.get("target")
    )
    if not channel_type or not channel_id:
        return {}
    principal_id = get_or_create_principal_for_channel(
        str(channel_type), str(channel_id)
    )
    if not principal_id:
        return {}
    return {
        "locale": get_with_fallback(
            principal_id, "locale", settings.get_default_locale()
        ),
        "tone": get_with_fallback(principal_id, "tone", settings.get_tone()),
        "address_style": get_with_fallback(
            principal_id, "address_style", settings.get_address_style()
        ),
        "reminders.relay_style": get_with_fallback(
            principal_id, "reminders.relay_style", True
        ),
    }
