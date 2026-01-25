import time

from core.context.providers.system_provider import get_system_context


def test_get_system_context_keys_and_values():
    context = get_system_context()

    assert context["atrium_status"] == "online"
    assert context["uptime_seconds"] >= 0
    assert isinstance(context["started_at_epoch"], int)
    assert context["started_at_epoch"] <= int(time.time())
