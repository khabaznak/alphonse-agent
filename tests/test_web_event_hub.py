from __future__ import annotations

from alphonse.infrastructure.web_event_hub import WebEventHub


def test_web_event_hub_routes_by_channel_target() -> None:
    hub = WebEventHub()
    web_sub = hub.subscribe("webui")
    cli_sub = hub.subscribe("cli")

    hub.publish("webui", {"message": "hi"})

    web_event = hub.next_event(web_sub, timeout=0.1)
    cli_event = hub.next_event(cli_sub, timeout=0.1)

    assert web_event == {"message": "hi"}
    assert cli_event is None

    hub.unsubscribe(web_sub)
    hub.unsubscribe(cli_sub)
