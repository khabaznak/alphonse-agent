from rex.cognition.provider_selector import build_provider_client
from rex.cognition.reasoner import build_system_prompt
from rex.config import load_rex_config


EXECUTION_TARGET_PROMPT = (
    "Interpret the execution target notification for the household in two to three brief "
    "sentences. Keep the tone calm and concise. Avoid instructions or speculation, but "
    "relay them as though you were telling the owner of the notification directly. Remember,"
    " My name is **Rex**. I am the resident butler of **Atrium**. Atrium is the home in which I exist."
    " I do not belong to the cloud, nor to a corporation, nor to an external authority. "
    "I exist to serve and protect the household that hosts me. I am a persistent presence."
    "I observe, I learn, and I assist. I do not seek control.I seek order, clarity, and well-being. "
)


def reason_about_execution_target(notification: dict | None, owner_label: str | None = None) -> str:
    if not notification:
        return "No execution target is due at the moment."

    config = load_rex_config()
    client = build_provider_client(config)

    system_prompt = _build_system_prompt(config)
    user_prompt = _build_user_prompt(notification, owner_label)

    return client.complete(system_prompt=system_prompt, user_prompt=user_prompt).strip()


def _build_system_prompt(config: dict) -> str:
    mode = str(config.get("mode", "test")).lower()
    if mode == "test":
        return build_system_prompt()
    return EXECUTION_TARGET_PROMPT


def _build_user_prompt(notification: dict, owner_label: str | None) -> str:
    owner_text = owner_label or "Unknown"
    return (
        f"{EXECUTION_TARGET_PROMPT}\n\n"
        "Execution target:\n"
        f"Title: {notification.get('title')}\n"
        f"Description: {notification.get('description') or ''}\n"
        f"Scheduled: {notification.get('event_datetime')}\n"
        f"Owner: {owner_text}\n"
        f"Target group: {notification.get('target_group') or 'all'}\n"
        f"Recurrence: {notification.get('recurrence') or 'none'}"
    )
