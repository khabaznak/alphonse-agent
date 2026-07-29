from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent import identity as users_store
from alphonse.agent.cognition.preferences.store import set_user_preference
from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.io import IntegrationIdentity
from alphonse.agent_v2.core.io import OutboundSelector
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.io import V2IdentityResolver
from alphonse.agent_v2.core.io import build_outbox_delivery_sink
from alphonse.agent_v2.core.io import channel_metadata
from alphonse.agent_v2.core.io import resolve_provider_user_mapping
from alphonse.agent_v2.core.io import upsert_provider_user_mapping
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native.ask_question import execute_ask_question


def test_outbox_selector_isolates_tui_messages() -> None:
    store = SQLiteOutboundStore()
    store.enqueue(
        address=ChannelAddress(
            integration_id="tui",
            provider_key="tui",
            channel_target="alex",
            alphonse_user_id="alex",
        ),
        message="hello tui",
    )
    store.enqueue(
        address=ChannelAddress(
            integration_id="telegram-home",
            provider_key="telegram",
            channel_target="123",
            alphonse_user_id="alex",
        ),
        message="hello telegram",
    )

    selected = store.claim_next(OutboundSelector(integration_id="tui", channel_target="alex"))

    assert selected is not None
    assert selected.message == "hello tui"
    assert store.list_pending(OutboundSelector(integration_id="telegram-home"))[0].message == "hello telegram"


def test_outbox_promotes_project_id_and_filters_it_directly() -> None:
    store = SQLiteOutboundStore()
    address = ChannelAddress(integration_id="desktop", provider_key="tui", channel_target="alex", alphonse_user_id="alex")
    alpha = store.enqueue(address=address, message="Alpha", project_id="alpha")
    store.enqueue(address=address, message="Beta", metadata={"project_id": "beta"})

    selected = store.list(OutboundSelector(integration_id="desktop", project_id="alpha"))

    assert [item.outbox_message_id for item in selected] == [alpha.outbox_message_id]
    assert selected[0].project_id == "alpha"
    assert store.list(OutboundSelector(project_id="beta"))[0].project_id == "beta"


def test_outbox_project_migration_backfills_metadata_idempotently(tmp_path: Path) -> None:
    db_path = tmp_path / "outbox.sqlite3"
    store = SQLiteOutboundStore(db_path)
    message = store.enqueue(
        address=ChannelAddress(integration_id="desktop", provider_key="tui", channel_target="alex", alphonse_user_id="alex"),
        message="Migrated",
        metadata={"project_id": "alpha"},
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_v2_outbox_project")
        conn.execute("ALTER TABLE v2_outbox DROP COLUMN project_id")

    migrated = SQLiteOutboundStore(db_path)
    restarted = SQLiteOutboundStore(db_path)

    assert migrated.get(message.outbox_message_id).project_id == "alpha"
    assert restarted.get(message.outbox_message_id).project_id == "alpha"


def test_outbox_retries_failed_delivery_before_terminal_failure() -> None:
    store = SQLiteOutboundStore()
    store.enqueue(
        address=ChannelAddress(
            integration_id="telegram-home",
            provider_key="telegram",
            channel_target="123",
            alphonse_user_id="u-alex",
        ),
        message="retry me",
    )

    claimed = store.claim_next(OutboundSelector(integration_id="telegram-home"))
    assert claimed is not None
    assert store.mark_failed(claimed.outbox_message_id, error="temporary", retry_after_seconds=0) is True
    retrying = store.list(OutboundSelector(integration_id="telegram-home", status="retry_wait"))
    assert retrying[0].last_error == "temporary"

    reclaimed = store.claim_next(OutboundSelector(integration_id="telegram-home"))
    assert reclaimed is not None
    assert reclaimed.outbox_message_id == claimed.outbox_message_id
    assert reclaimed.attempt_count == 2


def test_outbox_reclaims_expired_delivery_claims() -> None:
    store = SQLiteOutboundStore()
    store.enqueue(
        address=ChannelAddress(
            integration_id="telegram-home",
            provider_key="telegram",
            channel_target="123",
            alphonse_user_id="u-alex",
        ),
        message="recover delivery",
    )
    claimed = store.claim_next(
        OutboundSelector(integration_id="telegram-home"),
        lease_owner="worker-1",
        lease_seconds=1,
    )
    assert claimed is not None

    reclaimed_count = store.reclaim_expired(now=datetime.now(timezone.utc) + timedelta(seconds=2))
    assert reclaimed_count == 1
    reclaimed = store.claim_next(OutboundSelector(integration_id="telegram-home"), lease_owner="worker-2")

    assert reclaimed is not None
    assert reclaimed.outbox_message_id == claimed.outbox_message_id


def test_snapshot_projection_routes_bash_stdout_to_integration_outbox() -> None:
    from alphonse.agent_v2.core.core import ImprovementPhase, StateSnapshot
    from alphonse.agent_v2.core.io import channel_metadata
    from alphonse.agent_v2.core.io import project_snapshot_to_outbox

    snapshot = StateSnapshot(
        phase=ImprovementPhase.ACT,
        metadata={
            "task_state": {
                "user": "u-alex",
                "correlation_id": "telegram-home:55",
                "metadata": {
                    "channel": channel_metadata(
                        integration_id="telegram-home",
                        provider_key="telegram",
                        channel_target="999",
                        provider_message_id="55",
                        alphonse_user_id="u-alex",
                    )
                },
                "plan_json": [
                    {
                        "tool_id": "native.bash",
                        "execution": {"status": "success", "result": {"stdout": "10:42"}},
                    }
                ],
            }
        },
    )
    outbox = SQLiteOutboundStore()

    projected = project_snapshot_to_outbox(snapshot=snapshot, outbox=outbox)

    assert projected is not None
    assert projected.integration_id == "telegram-home"
    assert projected.message == "10:42"


def test_snapshot_projection_confirms_successful_artifact_to_origin_channel() -> None:
    from alphonse.agent_v2.core.core import ImprovementPhase, StateSnapshot
    from alphonse.agent_v2.core.io import channel_metadata, project_snapshot_to_outbox

    snapshot = StateSnapshot(
        phase=ImprovementPhase.ACT,
        metadata={"task_state": {"user": "u-alex", "metadata": {"channel": channel_metadata(integration_id="telegram-home", provider_key="telegram", channel_target="999", provider_message_id="56", alphonse_user_id="u-alex")}, "plan_json": [{"tool_id": "artifact.todo-list", "tool_name": "Add to TODO list", "execution": {"status": "success", "result": {"added": "Remember to pack your goggles"}}}]}},
    )

    projected = project_snapshot_to_outbox(snapshot=snapshot, outbox=SQLiteOutboundStore())

    assert projected is not None
    assert projected.integration_id == "telegram-home"
    assert projected.message == "Completed Add to TODO list."


def test_scheduled_task_final_response_is_copied_to_preferred_channel(tmp_path: Path) -> None:
    from alphonse.agent_v2.core.core import ImprovementPhase, StateSnapshot
    from alphonse.agent_v2.core.io import channel_metadata, project_snapshot_to_outbox
    from alphonse.agent_v2.users import V2UserStore

    users = V2UserStore(tmp_path / "users.sqlite3")
    alex = users.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    users.bind_address(
        user_id=alex.user_id,
        integration_id="telegram-home",
        provider_key="telegram",
        provider_user_id="123",
        channel_target="123",
        is_preferred=False,
    )
    users.bind_address(
        user_id=alex.user_id,
        integration_id="teams-work",
        provider_key="teams",
        provider_user_id="teams-alex",
        channel_target="teams-alex",
        is_preferred=True,
    )
    resolver = V2IdentityResolver(
        (IntegrationIdentity("telegram-home", "telegram"), IntegrationIdentity("teams-work", "teams")),
        user_store=users,
    )
    snapshot = StateSnapshot(
        phase=ImprovementPhase.ACT,
        metadata={
            "task_state": {
                "user": alex.user_id,
                "correlation_id": "scheduled:1",
                "metadata": {
                    "scheduled_task_id": "scheduled_task_1",
                    "scheduled_run_id": "scheduled_run_1",
                    "occurrence_key": "scheduled_task_1:scheduled_run_1",
                    "channel": channel_metadata(
                        integration_id="telegram-home",
                        provider_key="telegram",
                        channel_target="123",
                        alphonse_user_id=alex.user_id,
                    ),
                },
                "plan_json": [{"tool_id": "native.respond", "execution": {"status": "success", "result": {"message": "Time to stretch."}}}],
            }
        },
    )
    outbox = SQLiteOutboundStore()

    projected = project_snapshot_to_outbox(
        snapshot=snapshot,
        outbox=outbox,
        identity_resolver=resolver,
        mirror_automation_messages_to_preferred_channel=True,
    )

    assert projected is not None
    assert projected.integration_id == "telegram-home"
    copies = outbox.list(OutboundSelector(status=None))
    assert {(item.integration_id, item.channel_target) for item in copies} == {
        ("telegram-home", "123"),
        ("teams-work", "teams-alex"),
    }
    copy = next(item for item in copies if item.integration_id == "teams-work")
    assert copy.metadata["automation_preferred_channel_copy"] is True
    assert "occurrence_key" not in copy.metadata


def test_scheduled_task_copy_is_not_duplicated_for_matching_preferred_destination(tmp_path: Path) -> None:
    from alphonse.agent_v2.core.core import ImprovementPhase, StateSnapshot
    from alphonse.agent_v2.core.io import channel_metadata, project_snapshot_to_outbox
    from alphonse.agent_v2.users import V2UserStore

    users = V2UserStore(tmp_path / "users.sqlite3")
    alex = users.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    users.bind_address(user_id=alex.user_id, integration_id="telegram-home", provider_key="telegram", provider_user_id="123", channel_target="123")
    resolver = V2IdentityResolver((IntegrationIdentity("telegram-home", "telegram"),), user_store=users)
    snapshot = StateSnapshot(
        phase=ImprovementPhase.ACT,
        metadata={"task_state": {"user": alex.user_id, "metadata": {"scheduled_task_id": "scheduled_task_1", "channel": channel_metadata(integration_id="telegram-home", provider_key="telegram", channel_target="123", alphonse_user_id=alex.user_id)}, "plan_json": [{"tool_id": "native.respond", "execution": {"status": "success", "result": {"message": "One copy only."}}}]}},
    )
    outbox = SQLiteOutboundStore()

    project_snapshot_to_outbox(
        snapshot=snapshot,
        outbox=outbox,
        identity_resolver=resolver,
        mirror_automation_messages_to_preferred_channel=True,
    )

    assert len(outbox.list(OutboundSelector(status=None))) == 1


def test_identity_resolver_maps_inbound_and_preferred_outbound_across_providers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    _insert_channel(db_path, channel_id=88, channel_key="teams")
    users_store.upsert_user({"user_id": "u-alex", "display_name": "Alex", "is_active": True})
    users_store.upsert_user({"user_id": "u-gaby", "display_name": "Gaby", "is_active": True})
    resolvers.upsert_service_resolver(
        user_id="u-alex",
        service_id=2,
        service_user_id="123",
        is_active=True,
    )
    resolvers.upsert_service_resolver(
        user_id="u-gaby",
        service_id=88,
        service_user_id="98EBF",
        is_active=True,
    )
    set_user_preference("u-gaby", "preferred_communication_channel", "teams")
    resolver = V2IdentityResolver(
        (
            IntegrationIdentity("telegram-home", "telegram"),
            IntegrationIdentity("teams-work", "teams"),
        )
    )

    inbound = resolver.resolve_inbound_user(integration_id="telegram-home", provider_user_id="123")
    outbound = resolver.resolve_outbound_address(alphonse_user_id="u-gaby")

    assert inbound.resolved is True
    assert inbound.alphonse_user_id == "u-alex"
    assert outbound.resolved is True
    assert outbound.address is not None
    assert outbound.address.integration_id == "teams-work"
    assert outbound.address.channel_target == "98EBF"


def test_identity_resolver_reports_missing_mapping(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    resolver = V2IdentityResolver((IntegrationIdentity("telegram-home", "telegram"),))

    result = resolver.resolve_inbound_user(integration_id="telegram-home", provider_user_id="missing")

    assert result.resolved is False
    assert result.reason == "user_mapping_not_found"


def test_v2_identity_mapping_helper_creates_user_and_provider_mapping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    upsert_provider_user_mapping(
        alphonse_user_id="u-alex",
        provider_key="telegram",
        provider_user_id="123",
        display_name="Alex",
    )

    assert users_store.get_user("u-alex")["display_name"] == "Alex"
    assert resolve_provider_user_mapping(alphonse_user_id="u-alex", provider_key="telegram") == "123"
    resolver = V2IdentityResolver((IntegrationIdentity("telegram-home", "telegram"),))
    inbound = resolver.resolve_inbound_user(integration_id="telegram-home", provider_user_id="123")
    assert inbound.resolved is True
    assert inbound.alphonse_user_id == "u-alex"


def test_v2_identity_mapping_helper_rejects_provider_user_conflict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    upsert_provider_user_mapping(
        alphonse_user_id="u-alex",
        provider_key="telegram",
        provider_user_id="123",
        display_name="Alex",
    )

    try:
        upsert_provider_user_mapping(
            alphonse_user_id="u-gaby",
            provider_key="telegram",
            provider_user_id="123",
            display_name="Gaby",
        )
    except ValueError as exc:
        assert str(exc) == "provider_user_already_mapped"
    else:
        raise AssertionError("expected provider_user_already_mapped")


def test_delegated_question_routes_to_respondent_preferred_integration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    _insert_channel(db_path, channel_id=88, channel_key="teams")
    users_store.upsert_user({"user_id": "u-alex", "display_name": "Alex", "is_active": True})
    users_store.upsert_user({"user_id": "u-gaby", "display_name": "Gaby", "is_active": True})
    resolvers.upsert_service_resolver(user_id="u-gaby", service_id=88, service_user_id="98EBF")
    set_user_preference("u-gaby", "preferred_communication_channel", "teams")
    outbox = SQLiteOutboundStore()
    resolver = V2IdentityResolver(
        (
            IntegrationIdentity("telegram-home", "telegram"),
            IntegrationIdentity("teams-work", "teams"),
        )
    )
    task = TaskState(
        task_id="task-1",
        goal="Ask Gaby",
        user="u-alex",
        correlation_id="corr-1",
        metadata={
            "channel": channel_metadata(
                integration_id="telegram-home",
                provider_key="telegram",
                channel_target="123",
                provider_user_id="123",
                alphonse_user_id="u-alex",
            )
        },
    )
    store = SQLiteQuestionStore()

    result = execute_ask_question(
        {
            "question": "Can you review this?",
            "question_kind": "open_text",
            "respondent_user_id": "u-gaby",
        },
        context=ToolExecutionContext(
            task=task,
            messages=InMemoryMessageQueue(),
            question_store=store,
            delivery_sink=build_outbox_delivery_sink(outbox=outbox, identity_resolver=resolver),
        ),
    )

    assert result["waiting_for_answer"] is True
    outbound = outbox.claim_next(OutboundSelector(integration_id="teams-work", channel_target="98EBF"))
    assert outbound is not None
    assert outbound.kind == "question"
    assert outbound.audience_user_id == "u-gaby"
    assert outbound.message == "Can you review this?"
    question = store.get_question(result["question_interrupt"]["question_id"])
    assert question is not None
    assert question.metadata["delivery"]["integration_id"] == "teams-work"


def test_unresolved_delegated_question_notifies_originator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "u-alex", "display_name": "Alex", "is_active": True})
    users_store.upsert_user({"user_id": "u-gaby", "display_name": "Gaby", "is_active": True})
    outbox = SQLiteOutboundStore()
    task = TaskState(
        task_id="task-1",
        goal="Ask Gaby",
        user="u-alex",
        correlation_id="corr-1",
        metadata={
            "channel": channel_metadata(
                integration_id="telegram-home",
                provider_key="telegram",
                channel_target="123",
                provider_user_id="123",
                alphonse_user_id="u-alex",
            )
        },
    )

    result = execute_ask_question(
        {
            "question": "Can you review this?",
            "question_kind": "open_text",
            "respondent_user_id": "u-gaby",
        },
        context=ToolExecutionContext(
            task=task,
            messages=InMemoryMessageQueue(),
            question_store=SQLiteQuestionStore(),
            delivery_sink=build_outbox_delivery_sink(
                outbox=outbox,
                identity_resolver=V2IdentityResolver((IntegrationIdentity("telegram-home", "telegram"),)),
            ),
        ),
    )

    assert result["delivery_result"]["unresolved"] is True
    outbound = outbox.claim_next(OutboundSelector(integration_id="telegram-home", channel_target="123"))
    assert outbound is not None
    assert outbound.kind == "identity_resolution"
    assert "map user u-gaby" in outbound.message


def _insert_channel(db_path: Path, *, channel_id: int, channel_key: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO channels (
              channel_id, channel_key, provider, channel_type, raw_user_key_field, name, description
            ) VALUES (?, ?, ?, 'interactive', 'user_id', ?, ?)
            """,
            (channel_id, channel_key, channel_key, channel_key.title(), f"{channel_key.title()} delivery"),
        )
