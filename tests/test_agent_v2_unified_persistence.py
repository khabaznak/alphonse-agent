from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import EMPTY_MARKDOWN, TaskState
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.database import DEFAULT_BUSY_TIMEOUT_MS, connect_database, default_database_path
from alphonse.agent_v2.retention import prune_operational_data
from alphonse.agent_v2.storage_migration import migrate_legacy_databases
from alphonse.agent_v2.users import V2UserStore


def test_shared_database_configuration_and_defaults(monkeypatch, tmp_path) -> None:
    database = tmp_path / "alphonse-v2.sqlite3"
    monkeypatch.setenv("ALPHONSE_V2_DB_PATH", str(database))
    monkeypatch.setenv("ALPHONSE_V2_USERS_DB_PATH", str(tmp_path / "legacy-users.sqlite3"))

    stores = (
        V2UserStore.default(),
        SQLiteQuestionStore.default(),
        SQLiteOutboundStore.default(),
        SQLiteConversationStore.default(),
    )

    assert default_database_path() == database
    assert {store.db_path for store in stores} == {str(database)}
    with connect_database(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == DEFAULT_BUSY_TIMEOUT_MS
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_schema_migrations'"
        ).fetchone()


def test_legacy_migration_is_backed_up_validated_and_idempotent(tmp_path) -> None:
    source = tmp_path / "v2-users.sqlite3"
    question_source = tmp_path / "v2-questions.sqlite3"
    outbox_source = tmp_path / "v2-outbox.sqlite3"
    target = tmp_path / "alphonse-v2.sqlite3"
    users = V2UserStore(source)
    users.set_users_root(tmp_path / "users")
    users.create_user(display_name="Alex", role="admin", user_id="alex")
    questions = SQLiteQuestionStore(question_source)
    question = questions.create_question(
        task=TaskState(task_id="parent-task", user="alex", project_id="alpha", goal="Ask Gaby"),
        question="Which option?",
        respondent_user_id="gaby",
    )
    questions.bind_delivery_metadata(
        question_id=question.question_id,
        metadata={"provider_message_id": "question-provider-id"},
    )
    outbox = SQLiteOutboundStore(outbox_source)
    delivery = outbox.enqueue(
        address=ChannelAddress("telegram", "telegram", "chat-1", alphonse_user_id="alex"),
        message="Delivery",
        correlation_id="correlation-one",
        project_id="alpha",
    )
    outbox.mark_delivered(delivery.outbox_message_id, provider_message_id="delivery-provider-id")
    sources = [source, question_source, outbox_source]

    first = migrate_legacy_databases(
        target=target,
        sources=sources,
        backup_root=tmp_path / "backups",
    )
    second = migrate_legacy_databases(
        target=target,
        sources=sources,
        backup_root=tmp_path / "backups",
    )

    assert first["status"] == "migrated"
    assert second["status"] == "already_applied"
    assert V2UserStore(target).get_user("alex") is not None
    migrated_question = SQLiteQuestionStore(target).get_question(question.question_id)
    assert migrated_question is not None
    assert migrated_question.status == "pending"
    assert migrated_question.metadata["delivery"]["provider_message_id"] == "question-provider-id"
    assert SQLiteQuestionStore(target).load_task_checkpoint("parent-task") is not None
    with connect_database(target) as connection:
        dependency = connection.execute(
            "SELECT * FROM v2_task_dependencies WHERE question_id=?",
            (question.question_id,),
        ).fetchone()
    assert dependency is not None
    migrated_delivery = SQLiteOutboundStore(target).get(delivery.outbox_message_id)
    assert migrated_delivery is not None
    assert migrated_delivery.status == "delivered"
    assert migrated_delivery.provider_message_id == "delivery-provider-id"
    backup_directories = list((tmp_path / "backups").glob("pre-unified-*"))
    assert len(backup_directories) == 1
    backup_directory = backup_directories[0]
    assert {item.name for item in sources}.issubset({item.name for item in backup_directory.iterdir()})
    with connect_database(target) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM v2_schema_migrations").fetchone()[0] == 2


def test_failed_migration_does_not_modify_legacy_database(tmp_path) -> None:
    source = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE v2_unknown (id TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO v2_unknown VALUES ('one','still here')")

    with pytest.raises(RuntimeError, match="unified_schema_missing_table"):
        migrate_legacy_databases(
            target=tmp_path / "unified.sqlite3",
            sources=[source],
            backup_root=tmp_path / "backups",
        )

    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT value FROM v2_unknown WHERE id='one'").fetchone()[0] == "still here"


def test_checkpoint_omits_ledger_and_resumes_actionable_state(tmp_path) -> None:
    store = SQLiteQuestionStore(tmp_path / "unified.sqlite3")
    ledger = "# Memory Ledger\n" + ("important context " * 100_000)
    task = TaskState(
        task_id="task-one",
        user="alex",
        project_id="project-one",
        goal="Continue the work",
        facts_md="- durable fact",
        conversation_history_md=ledger,
    )
    question = store.create_question(task=task, question="Continue?", kind="yes_no")

    with connect_database(store.db_path) as connection:
        payload = connection.execute(
            "SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task-one'"
        ).fetchone()[0]
    restored = store.load_task_checkpoint("task-one")
    answered = store.route_answer(
        respondent_user_id="alex",
        question_id=question.question_id,
        text="yes",
    )

    assert len(payload) < 20_000
    assert "important context" not in payload
    assert restored is not None
    assert restored.conversation_history_md == EMPTY_MARKDOWN
    assert restored.facts_md == "- durable fact"
    assert answered.resumed_task is not None


def test_retention_removes_only_old_terminal_operational_rows(tmp_path) -> None:
    database = tmp_path / "unified.sqlite3"
    questions = SQLiteQuestionStore(database)
    outbox = SQLiteOutboundStore(database)
    conversations = SQLiteConversationStore(database)
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()

    completed_task = TaskState(task_id="completed", user="alex", goal="Done")
    completed_question = questions.create_question(task=completed_task, question="Done?", kind="yes_no")
    questions.route_answer(respondent_user_id="alex", question_id=completed_question.question_id, text="yes")
    questions.mark_task_checkpoint_terminal("completed", status="done")
    pending_question = questions.create_question(
        task=TaskState(task_id="pending", user="alex", goal="Wait"),
        question="Still pending?",
    )
    delivered = outbox.enqueue(
        address=ChannelAddress("desktop", "tui", "alex", alphonse_user_id="alex"),
        message="Delivered",
    )
    outbox.mark_delivered(delivered.outbox_message_id)
    conversations.record(
        owner_user_id="alex",
        project_id="project",
        role="assistant",
        content="Retain forever",
        source="desktop",
        source_message_id="conversation-one",
    )
    with connect_database(database) as connection:
        connection.execute(
            "UPDATE v2_questions SET updated_at=? WHERE question_id=?",
            (old, completed_question.question_id),
        )
        connection.execute(
            "UPDATE v2_task_checkpoints SET updated_at=? WHERE task_id='completed'",
            (old,),
        )
        connection.execute(
            "UPDATE v2_outbox SET updated_at=? WHERE outbox_message_id=?",
            (old, delivered.outbox_message_id),
        )

    deleted = prune_operational_data(database)

    assert deleted["questions"] == 1
    assert deleted["task_checkpoints"] == 1
    assert deleted["outbound"] == 1
    assert questions.get_question(completed_question.question_id) is None
    assert questions.get_question(pending_question.question_id) is not None
    assert questions.load_task_checkpoint("pending") is not None
    assert len(conversations.list(owner_user_id="alex", project_id="project")) == 1


def test_concurrent_store_writers_share_wal_database(tmp_path) -> None:
    database = tmp_path / "unified.sqlite3"
    SQLiteConversationStore(database)
    SQLiteOutboundStore(database)

    def write(index: int) -> None:
        if index % 2:
            SQLiteConversationStore(database).record(
                owner_user_id=f"user-{index % 3}",
                project_id=f"project-{index % 2}",
                role="user",
                content=f"message {index}",
                source="test",
                source_message_id=f"inbound:{index}",
            )
        else:
            SQLiteOutboundStore(database).enqueue(
                address=ChannelAddress("desktop", "tui", f"user-{index % 3}"),
                message=f"response {index}",
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(80)))

    with connect_database(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM v2_conversation_events").fetchone()[0] == 40
        assert connection.execute("SELECT COUNT(*) FROM v2_outbox").fetchone()[0] == 40
