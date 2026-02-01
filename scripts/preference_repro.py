from __future__ import annotations

from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_with_fallback,
    set_preference,
)
from alphonse.config import settings


def main() -> None:
    channel_type = "telegram"
    channel_id = "8553589429"

    principal_id = get_or_create_principal_for_channel(channel_type, channel_id)
    if not principal_id:
        raise RuntimeError("Failed to resolve principal")

    set_preference(principal_id, "locale", "en-US", source="user")
    locale = get_with_fallback(principal_id, "locale", settings.get_default_locale())

    print("principal_id:", principal_id)
    print("locale:", locale)

    effective_locale = get_with_fallback(
        principal_id, "locale", settings.get_default_locale()
    )
    print("effective_locale:", effective_locale)


if __name__ == "__main__":
    main()
