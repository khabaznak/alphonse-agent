from __future__ import annotations

from pathlib import Path

from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.io import CommunicationRouter
from alphonse.agent_v2.core.io import IntegrationIdentity
from alphonse.agent_v2.core.io import OutboundSelector
from alphonse.agent_v2.core.io import SQLiteCommunicationThreadStore
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.io import V2IdentityResolver
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.services.project_sessions import ProjectInboundRouter
from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore
from alphonse.agent_v2.users import V2UserStore


def _router(tmp_path: Path):
    users = V2UserStore(tmp_path / "users.sqlite3")
    admin = users.onboard(display_name="Steve", users_root=tmp_path / "profiles")
    gaby = users.create_user(display_name="Gabriela Villasana Rodríguez")
    users.set_aliases(gaby.user_id, ["Gaby"])
    users.bind_address(user_id=gaby.user_id, integration_id="telegram-home", provider_key="telegram", provider_user_id="222", channel_target="222")
    outbox = SQLiteOutboundStore()
    router = CommunicationRouter(
        users=users,
        resolver=V2IdentityResolver((IntegrationIdentity("telegram-home", "telegram"),), user_store=users),
        outbox=outbox,
        threads=SQLiteCommunicationThreadStore(),
    )
    return router, outbox, admin, gaby


def test_deliver_to_user_uses_recipient_address_and_relays_explicit_reply(tmp_path: Path) -> None:
    router, outbox, admin, gaby = _router(tmp_path)
    origin = ChannelAddress("desktop", "tui", admin.user_id, admin.user_id, admin.user_id)

    result = router.deliver(sender_user_id=admin.user_id, origin=origin, recipient_reference="gaby", message="Would you like lunch at home?", expects_reply=True)

    outbound = outbox.claim_next(OutboundSelector(integration_id="telegram-home"))
    assert result["status"] == "queued"
    assert outbound is not None
    assert outbound.channel_target == "222"
    assert outbound.message == "Steve says: Would you like lunch at home?"
    router.threads.mark_delivered(outbound.outbox_message_id, "telegram-7")

    handled = router.relay_inbound(
        sender_user_id=gaby.user_id,
        address=ChannelAddress("telegram-home", "telegram", "222", gaby.user_id, "222", "telegram-8", "telegram-7"),
        text="Thank you Steve!",
    )

    reply = outbox.claim_next(OutboundSelector(integration_id="desktop", channel_target=admin.user_id))
    assert handled is True
    assert reply is not None
    assert reply.message == "Gabriela Villasana Rodríguez replied: Thank you Steve!"


def test_plain_reply_requires_exactly_one_open_thread_and_unknown_recipient_is_safe(tmp_path: Path) -> None:
    router, outbox, admin, gaby = _router(tmp_path)
    origin = ChannelAddress("desktop", "tui", admin.user_id, admin.user_id, admin.user_id)
    assert router.deliver(sender_user_id=admin.user_id, origin=origin, recipient_reference="Missing", message="Hello")["status"] == "recipient_not_found"
    result = router.deliver(sender_user_id=admin.user_id, origin=origin, recipient_reference="Gaby", message="Hello")
    outbound = outbox.claim_next(OutboundSelector(integration_id="telegram-home"))
    assert outbound is not None
    router.threads.mark_delivered(result["outbox_message_id"], "telegram-9")

    assert router.relay_inbound(sender_user_id=gaby.user_id, address=ChannelAddress("telegram-home", "telegram", "222", gaby.user_id, "222", "telegram-10"), text="Hi") is True


def test_recipient_resolution_supports_nickname_unaccented_name_and_rejects_ambiguity(tmp_path: Path) -> None:
    router, _outbox, admin, gaby = _router(tmp_path)
    origin = ChannelAddress("desktop", "tui", admin.user_id, admin.user_id, admin.user_id)

    assert router.deliver(sender_user_id=admin.user_id, origin=origin, recipient_reference="Gaby", message="Hi")["recipient_user_id"] == gaby.user_id
    assert router.deliver(sender_user_id=admin.user_id, origin=origin, recipient_reference="Gabriela", message="Hi")["recipient_user_id"] == gaby.user_id
    assert router.deliver(sender_user_id=admin.user_id, origin=origin, recipient_reference="Gabriela Villasana Rodriguez", message="Hi")["recipient_user_id"] == gaby.user_id

    gabriel = router.users.create_user(display_name="Gabriel Gómez")
    assert router.deliver(sender_user_id=admin.user_id, origin=origin, recipient_reference="Gab", message="Hi")["status"] == "recipient_ambiguous"
    assert gabriel.user_id


def test_inbound_router_consumes_linked_reply_before_queueing_capd_work(tmp_path: Path) -> None:
    router, outbox, admin, gaby = _router(tmp_path)
    origin = ChannelAddress("desktop", "tui", admin.user_id, admin.user_id, admin.user_id)
    sent = router.deliver(sender_user_id=admin.user_id, origin=origin, recipient_reference="Gaby", message="Are you home?", expects_reply=True)
    outbound = outbox.claim_next(OutboundSelector(integration_id="telegram-home"))
    assert outbound is not None
    router.threads.mark_delivered(sent["outbox_message_id"], "telegram-11")
    queue = InMemoryMessageQueue()
    inbound = ProjectInboundRouter(
        channel=CommunicationChannel(queue), outbox=outbox, projects=ProjectStore(), sessions=SQLiteProjectSessionStore(), communication_router=router,
    )

    result = inbound.ingest(prompt="Yes", user=gaby.user_id, integration_id="telegram-home", provider_key="telegram", provider_user_id="222", channel_target="222", provider_message_id="telegram-12", reply_to_provider_message_id="telegram-11")

    assert result.handled_command is True
    assert queue.size() == 0
    assert outbox.claim_next(OutboundSelector(integration_id="desktop", channel_target=admin.user_id)) is not None
