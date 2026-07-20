"""Deterministic person-to-person routing, independent of CAPD."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.identity import V2IdentityResolver
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore


@dataclass(frozen=True)
class CommunicationThread:
    thread_id: str
    sender_user_id: str
    recipient_user_id: str
    origin: ChannelAddress
    recipient: ChannelAddress
    outbox_message_id: str
    provider_message_id: str
    status: str
    expires_at: str


class SQLiteCommunicationThreadStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteCommunicationThreadStore":
        return cls(Path.home() / ".alphonse" / "v2-communication-threads.sqlite3")

    def create(self, *, sender_user_id: str, recipient_user_id: str, origin: ChannelAddress, recipient: ChannelAddress, outbox_message_id: str) -> CommunicationThread:
        now = _now()
        thread = CommunicationThread(str(uuid4()), sender_user_id, recipient_user_id, origin, recipient, outbox_message_id, "", "pending_delivery", (now + timedelta(days=7)).isoformat())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO v2_communication_threads (thread_id,sender_user_id,recipient_user_id,origin_json,recipient_json,outbox_message_id,provider_message_id,status,expires_at,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (thread.thread_id, thread.sender_user_id, thread.recipient_user_id, json.dumps(origin.to_dict(), sort_keys=True), json.dumps(recipient.to_dict(), sort_keys=True), thread.outbox_message_id, "", thread.status, thread.expires_at, now.isoformat(), now.isoformat()),
            )
        return thread

    def mark_delivered(self, outbox_message_id: str, provider_message_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE v2_communication_threads SET provider_message_id=?, status='open', updated_at=? WHERE outbox_message_id=? AND status='pending_delivery'", (str(provider_message_id), _now().isoformat(), str(outbox_message_id)))

    def mark_failed(self, outbox_message_id: str) -> CommunicationThread | None:
        with self._connect() as conn:
            conn.execute("UPDATE v2_communication_threads SET status='failed', updated_at=? WHERE outbox_message_id=? AND status IN ('pending_delivery','open')", (_now().isoformat(), str(outbox_message_id)))
        return self.by_outbox(outbox_message_id)

    def by_outbox(self, outbox_message_id: str) -> CommunicationThread | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_communication_threads WHERE outbox_message_id=?", (str(outbox_message_id),)).fetchone()
        return _thread(row)

    def consume_reply(self, *, sender_user_id: str, address: ChannelAddress) -> CommunicationThread | None:
        self.expire_due()
        explicit = str(address.reply_to_provider_message_id or "").strip()
        with self._connect() as conn:
            if explicit:
                rows = conn.execute("SELECT * FROM v2_communication_threads WHERE recipient_user_id=? AND provider_message_id=? AND status='open'", (sender_user_id, explicit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM v2_communication_threads WHERE recipient_user_id=? AND status='open' AND expires_at>?", (sender_user_id, _now().isoformat())).fetchall()
            matches = [item for item in (_thread(row) for row in rows) if item is not None and item.recipient.integration_id == address.integration_id and item.recipient.channel_target == address.channel_target]
            if len(matches) != 1:
                return None
            thread = matches[0]
            conn.execute("UPDATE v2_communication_threads SET status='replied', updated_at=? WHERE thread_id=? AND status='open'", (_now().isoformat(), thread.thread_id))
        return thread

    def expire_due(self) -> int:
        with self._connect() as conn:
            return conn.execute("UPDATE v2_communication_threads SET status='expired', updated_at=? WHERE status IN ('pending_delivery','open') AND expires_at<=?", (_now().isoformat(), _now().isoformat())).rowcount

    def _connect(self) -> sqlite3.Connection:
        if self._memory is not None:
            return self._memory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS v2_communication_threads (thread_id TEXT PRIMARY KEY, sender_user_id TEXT NOT NULL, recipient_user_id TEXT NOT NULL, origin_json TEXT NOT NULL, recipient_json TEXT NOT NULL, outbox_message_id TEXT NOT NULL UNIQUE, provider_message_id TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, expires_at TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_communication_threads_reply ON v2_communication_threads(recipient_user_id, provider_message_id, status)")


class CommunicationRouter:
    def __init__(self, *, users: Any, resolver: V2IdentityResolver, outbox: SQLiteOutboundStore, threads: SQLiteCommunicationThreadStore) -> None:
        self.users = users
        self.resolver = resolver
        self.outbox = outbox
        self.threads = threads

    def deliver(self, *, sender_user_id: str, origin: ChannelAddress, recipient_reference: str, message: str, expects_reply: bool = False) -> dict[str, Any]:
        recipient = self._resolve_user(recipient_reference)
        if recipient is None:
            return {"status": "recipient_not_found"}
        if recipient == "ambiguous":
            return {"status": "recipient_ambiguous"}
        resolved = self.resolver.resolve_outbound_address(alphonse_user_id=recipient)
        if not resolved.resolved or resolved.address is None:
            return {"status": "recipient_unreachable", "reason": resolved.reason}
        sender = self.users.get_user(sender_user_id)
        sender_name = str(getattr(sender, "display_name", "") or sender_user_id)
        outbound = self.outbox.enqueue(address=resolved.address, message=f"{sender_name} says: {message}", kind="person_message", audience_user_id=recipient, metadata={"communication": True, "expects_reply": bool(expects_reply)})
        thread = self.threads.create(sender_user_id=sender_user_id, recipient_user_id=recipient, origin=origin, recipient=resolved.address, outbox_message_id=outbound.outbox_message_id)
        return {"status": "queued", "thread_id": thread.thread_id, "outbox_message_id": outbound.outbox_message_id, "recipient_user_id": recipient}

    def relay_inbound(self, *, sender_user_id: str, address: ChannelAddress, text: str) -> bool:
        thread = self.threads.consume_reply(sender_user_id=sender_user_id, address=address)
        if thread is None:
            return False
        sender = self.users.get_user(sender_user_id)
        name = str(getattr(sender, "display_name", "") or sender_user_id)
        self.outbox.enqueue(address=thread.origin, message=f"{name} replied: {text}", kind="communication_reply", audience_user_id=thread.sender_user_id, reply_to_provider_message_id=thread.origin.provider_message_id, metadata={"communication_thread_id": thread.thread_id})
        return True

    def _resolve_user(self, reference: str) -> str | None:
        status, users = self.users.resolve_user_reference(reference)
        if status == "resolved":
            return users[0].user_id
        return "ambiguous" if status == "ambiguous" else None


def _thread(row: sqlite3.Row | None) -> CommunicationThread | None:
    if row is None:
        return None
    return CommunicationThread(str(row["thread_id"]), str(row["sender_user_id"]), str(row["recipient_user_id"]), ChannelAddress(**json.loads(str(row["origin_json"]))), ChannelAddress(**json.loads(str(row["recipient_json"]))), str(row["outbox_message_id"]), str(row["provider_message_id"]), str(row["status"]), str(row["expires_at"]))


def _now() -> datetime:
    return datetime.now(timezone.utc)
