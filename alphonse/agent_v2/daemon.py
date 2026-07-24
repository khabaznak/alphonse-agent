"""Foreground v2 daemon host."""

from __future__ import annotations

import signal
import fcntl
import os
import sqlite3
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.io import OutboundSelector
from alphonse.agent_v2.core.io import upsert_provider_user_mapping
from alphonse.agent_v2.core.io import project_snapshot_to_outbox
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.scheduled_tasks import schedule_summary
from alphonse.agent_v2.integrations import SQLiteIntegrationStore
from alphonse.agent_v2.inference_settings import SQLiteInferenceSettingsStore
from alphonse.agent_v2.ipc import V2DaemonClient
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.ipc import default_socket_path
from alphonse.agent_v2.runtime import V2RuntimeHost
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.runtime import start_runtime_integrations
from alphonse.agent_v2.runtime import stop_runtime_integrations
from alphonse.agent_v2.runtime import refresh_runtime_inference
from alphonse.agent_v2.runtime import refresh_runtime_identity_resolver
from alphonse.agent_v2.inference_settings import validate_and_save_inference_settings
from alphonse.agent_v2.services.project_sessions import ProjectSessionKey
from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore
from alphonse.agent_v2.agent_config import AgentConfigStore
from alphonse.agent_v2.interfaces.a2ui import ALPHONSE_DESKTOP_CATALOG_ID
from alphonse.agent_v2.interfaces.a2ui import A2UiAdapter
from alphonse.agent_v2.interfaces.a2ui import question_id_from_surface
from alphonse.agent_v2.interfaces.a2ui import surface_id_for_question
from alphonse.agent_v2.interfaces.ag_ui import AgUiAdapter
from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.web_tools_settings import WebToolsSettings
from alphonse.agent_v2.memory_settings import MemorySettings
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore
from alphonse.agent_v2.web_tools_settings import SQLiteWebToolsSettingsStore
from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore
from alphonse.agent_v2.core.tools.registry.native.media import verify_ocr, verify_stt, verify_tts
from alphonse.agent_v2.runtime import refresh_runtime_web_tools
from alphonse.agent_v2.runtime import refresh_runtime_media_tools
from alphonse.agent_v2.core.tools.registry.native.web import execute_web_fetch, execute_web_search
from alphonse.agent_v2.assets import SQLiteAssetStore
from alphonse.agent_v2.conversations import SQLiteConversationStore, legacy_ledger_events


@dataclass
class V2Daemon:
    runtime: V2RuntimeHost
    poll_interval_sec: float = 0.05
    inbound_max_attempts: int = 5

    def __post_init__(self) -> None:
        self.daemon_id = f"daemon-{uuid4().hex[:12]}"
        if hasattr(self.runtime.queue, "lease_owner"):
            self.runtime.queue.lease_owner = self.daemon_id
        self._stop = threading.Event()
        self._processor_thread: threading.Thread | None = None
        self._lock_file: Any | None = None
        self._lifecycle_lock = threading.RLock()
        self._stopped = False
        self._last_processor_error = ""
        self._active_work_lock = threading.RLock()
        self._active_work: dict[str, str] = {}
        self._activity_status: dict[str, str] = {"state": "idle", "updated_at": datetime.now(timezone.utc).isoformat()}
        self._event_lock = threading.RLock()
        self._activity_event_sequence = 0
        self._legacy_activity_cursor = 0
        self._activity_event_journal: list[dict[str, Any]] = []
        self._ui_event_lock = threading.RLock()
        self._ui_event_sequence = 0
        self._ui_event_journal: list[dict[str, Any]] = []
        self._desktop_capabilities: dict[str, set[str]] = {}
        self._desktop_surfaces: dict[tuple[str, str], set[str]] = {}
        self._ag_ui = AgUiAdapter(question_store=self.runtime.question_store)
        self._a2ui = A2UiAdapter()
        self.scheduler = ScheduledTaskWorker(
            store=self.runtime.schedule_store,
            messages=self.runtime.queue,
            worker_id=self.daemon_id,
            on_message_queued=lambda: None,
        )
        self.ipc = V2DaemonServer(self)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._stopped:
                raise RuntimeError("alphonse_v2_daemon_stopped")
            if self._processor_thread is not None and self._processor_thread.is_alive():
                return
            self._acquire_single_instance_lock()
        try:
            self._stop.clear()
            reclaim_expired = getattr(self.runtime.queue, "reclaim_expired", None)
            if callable(reclaim_expired):
                reclaim_expired()
            # Publish the health socket before optional providers initialize so
            # clients can distinguish a live daemon from a failed startup.
            self.ipc.start()
            self._align_configured_provider_addresses()
            start_runtime_integrations(
                self.runtime,
                on_outbox_delivered=self._on_outbox_delivered,
                on_outbox_failed=self._on_outbox_failed,
            )
            self.scheduler.start()
            self._processor_thread = threading.Thread(target=self._process_loop, name="alphonse-v2-core", daemon=True)
            self._processor_thread.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop.set()
        self.ipc.stop()
        self.scheduler.stop()
        stop_runtime_integrations(self.runtime)
        if self._processor_thread is not None and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)
        self._release_single_instance_lock()

    def restart_integrations(self) -> None:
        self._align_configured_provider_addresses()
        start_runtime_integrations(
            self.runtime,
            on_outbox_delivered=self._on_outbox_delivered,
            on_outbox_failed=self._on_outbox_failed,
        )

    def _align_configured_provider_addresses(self) -> None:
        records = self.runtime.integration_store.list_enabled()
        for record in records:
            same_provider = [item for item in records if item.provider_key == record.provider_key]
            if len(same_provider) == 1:
                self.runtime.user_store.align_provider_addresses(
                    provider_key=record.provider_key,
                    integration_id=record.integration_id,
                )

    def update_inference_settings(self, *, provider_key: str, model_id: str) -> dict[str, str]:
        settings = validate_and_save_inference_settings(
            self.runtime.inference_settings_store,
            provider_key=provider_key,
            model_id=model_id,
        )
        refresh_runtime_inference(self.runtime, settings)
        return settings.to_dict()

    def onboarding_status(self) -> dict[str, object]:
        return self.runtime.user_store.status()

    def onboard(self, *, display_name: str, users_root: str, import_v1: bool = False) -> dict[str, object]:
        store = self.runtime.user_store
        if import_v1:
            store.set_users_root(users_root)
            imported = store.import_v1()
            admin = store.admin_user()
            if admin is None:
                admin = store.onboard(display_name=display_name, users_root=users_root)
        else:
            admin = store.onboard(display_name=display_name, users_root=users_root)
            imported = {}
        self.runtime.user = admin.user_id
        migrated = self._migrate_legacy_local(admin.user_id)
        refresh_runtime_identity_resolver(self.runtime)
        return {"admin_user": admin.to_dict(), "migration": {**imported, **migrated}, "users_root": str(store.users_root())}

    def _migrate_legacy_local(self, admin_user_id: str) -> dict[str, int]:
        """Idempotently claim records created before canonical-user onboarding."""
        projects = self.runtime.project_store.migrate_owner("local", admin_user_id)
        sessions = self.runtime.project_session_store.migrate_user("local", admin_user_id)
        integrations = 0
        schedules = 0
        for record in self.runtime.integration_store.list():
            config = dict(record.config)
            if str(config.get("owner_user_id") or "") != "local":
                continue
            config["owner_user_id"] = admin_user_id
            self.runtime.integration_store.upsert(integration_id=record.integration_id, provider_key=record.provider_key, display_name=record.display_name, enabled=record.enabled, config=config, secrets=dict(record.secrets))
            integrations += 1
        if getattr(self.runtime.schedule_store, "db_path", ":memory:") != ":memory:":
            with sqlite3.connect(self.runtime.schedule_store.db_path) as conn:
                schedules = conn.execute("UPDATE v2_scheduled_tasks SET owner_user_id=? WHERE owner_user_id='local'", (admin_user_id,)).rowcount
        return {"local_projects_migrated": projects, "local_sessions_migrated": sessions, "local_integrations_migrated": integrations, "local_schedules_migrated": schedules}

    def current_user(self) -> dict[str, object]:
        admin = self.runtime.user_store.admin_user()
        return {"user": admin.to_dict() if admin else None, "onboarded": admin is not None}

    def _admin_user_id(self, requested: str = "") -> str:
        admin = self.runtime.user_store.admin_user()
        if admin is None:
            if self.runtime.user_store.is_ephemeral and str(requested or "").strip():
                return str(requested).strip()
            raise RuntimeError("v2_onboarding_required")
        return admin.user_id

    def list_users(self) -> list[dict[str, object]]:
        self.runtime.user_store.normalize_duplicate_addresses({record.integration_id for record in self.runtime.integration_store.list()})
        return [{**user.to_dict(), "addresses": [address.to_dict() for address in self.runtime.user_store.list_addresses(user.user_id)], "aliases": self.runtime.user_store.list_aliases(user.user_id)} for user in self.runtime.user_store.list_users()]

    def create_user(self, *, display_name: str, role: str = "member") -> dict[str, object]:
        return self.runtime.user_store.create_user(display_name=display_name, role=role).to_dict()

    def scheduled_tasks(
        self,
        *,
        actor_user_id: str = "",
        owner_user_id: str = "",
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        owner = self._scheduled_task_owner(actor_user_id=actor_user_id, requested_owner_user_id=owner_user_id)
        tasks = self.runtime.schedule_store.list_tasks(owner_user_id=owner, status=status, limit=limit)
        return [{**task.to_dict(), "latest_execution": self._latest_scheduled_execution(task.scheduled_task_id)} for task in tasks]

    def scheduled_task_executions(self, *, actor_user_id: str = "", scheduled_task_id: str, limit: int = 100) -> list[dict[str, object]]:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return [item.to_dict() for item in self.runtime.schedule_store.list_executions(scheduled_task_id=scheduled_task_id, limit=limit)]

    def update_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str, name: str, prompt: str) -> dict[str, object]:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return self.runtime.schedule_store.update_task(scheduled_task_id, name=name, prompt=prompt).to_dict()

    def pause_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str) -> dict[str, object]:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return self.runtime.schedule_store.pause_task(scheduled_task_id).to_dict()

    def resume_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str) -> dict[str, object]:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return self.runtime.schedule_store.resume_task(scheduled_task_id).to_dict()

    def cancel_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str) -> dict[str, object]:
        task = self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        if getattr(task, "status", "") not in {"active", "paused"}:
            raise ValueError("scheduled_task_not_cancellable")
        return self.runtime.schedule_store.cancel_task(scheduled_task_id).to_dict()

    def delete_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str) -> bool:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return self.runtime.schedule_store.delete_task(scheduled_task_id)

    def _scheduled_task_owner(self, *, actor_user_id: str, requested_owner_user_id: str) -> str:
        actor = str(actor_user_id or self._admin_user_id()).strip() or self._admin_user_id()
        if self.runtime.user_store.get_user(actor) is None:
            raise ValueError("scheduled_task_actor_not_found")
        requested = str(requested_owner_user_id or actor).strip() or actor
        if requested != actor and not self.runtime.user_store.is_admin(actor):
            raise PermissionError("scheduled_task_owner_forbidden")
        if self.runtime.user_store.get_user(requested) is None:
            raise KeyError("scheduled_task_owner_not_found")
        return requested

    def _scheduled_task_for_actor(self, *, actor_user_id: str, scheduled_task_id: str) -> object:
        task = self.runtime.schedule_store.get_task(scheduled_task_id)
        if task is None:
            raise KeyError(f"scheduled_task_not_found: {scheduled_task_id}")
        owner = self._scheduled_task_owner(actor_user_id=actor_user_id, requested_owner_user_id=task.owner_user_id)
        if task.owner_user_id != owner:
            raise PermissionError("scheduled_task_owner_forbidden")
        return task

    def _latest_scheduled_execution(self, scheduled_task_id: str) -> dict[str, object] | None:
        executions = self.runtime.schedule_store.list_executions(scheduled_task_id=scheduled_task_id, limit=1)
        return executions[0].to_dict() if executions else None

    def update_user(self, user_id: str, **values: Any) -> dict[str, object]:
        return self.runtime.user_store.update_user(user_id, display_name=values.get("display_name"), role=values.get("role"), is_active=values.get("is_active")).to_dict()

    def delete_user(self, user_id: str, *, confirmation: str) -> dict[str, int]:
        self._admin_user_id()
        user = self.runtime.user_store.get_user(user_id)
        if user is None:
            raise KeyError("user_not_found")
        if str(confirmation or "") != user.user_id:
            raise ValueError("delete_confirmation_must_match_user_id")
        roots = self.runtime.project_store.delete_owned_by(user.user_id)
        self.runtime.project_session_store.delete_user(user.user_id)
        for root in roots:
            path = Path(root)
            if path.exists() and self.runtime.user_store.users_root() in path.parents:
                shutil.rmtree(path)
        schedules = self._delete_user_schedule_data(user.user_id)
        questions = self._delete_user_question_data(user.user_id)
        inbound = self._delete_user_inbound_data(user.user_id)
        outbound = self._delete_user_outbound_data(user.user_id)
        deleted = self.runtime.user_store.delete_user(user.user_id)
        return {"deleted": int(deleted), "projects": len(roots), "schedules": schedules, "questions": questions, "inbound": inbound, "outbound": outbound}

    def _delete_user_schedule_data(self, user_id: str) -> int:
        store = self.runtime.schedule_store
        with store._connect() as conn:
            ids = [str(row[0]) for row in conn.execute("SELECT scheduled_task_id FROM v2_scheduled_tasks WHERE owner_user_id=?", (user_id,)).fetchall()]
            if ids:
                conn.executemany("DELETE FROM v2_scheduled_task_executions WHERE scheduled_task_id=?", [(task_id,) for task_id in ids])
            return conn.execute("DELETE FROM v2_scheduled_tasks WHERE owner_user_id=?", (user_id,)).rowcount

    def _delete_user_question_data(self, user_id: str) -> int:
        store = self.runtime.question_store
        with store._connect() as conn:
            rows = conn.execute("SELECT question_id FROM v2_questions WHERE respondent_user_id=? OR originator_user_id=?", (user_id, user_id)).fetchall()
            ids = [str(row[0]) for row in rows]
            if ids:
                conn.executemany("DELETE FROM v2_task_dependencies WHERE question_id=?", [(question_id,) for question_id in ids])
            conn.execute("DELETE FROM v2_task_checkpoints WHERE owner_id=?", (user_id,))
            return conn.execute("DELETE FROM v2_questions WHERE respondent_user_id=? OR originator_user_id=?", (user_id, user_id)).rowcount

    def _delete_user_inbound_data(self, user_id: str) -> int:
        queue = self.runtime.queue
        connect = getattr(queue, "_connect", None)
        if not callable(connect):
            return 0
        with connect() as conn:
            return conn.execute("DELETE FROM v2_inbound_messages WHERE user_id=?", (user_id,)).rowcount

    def _delete_user_outbound_data(self, user_id: str) -> int:
        with self.runtime.outbox._connect() as conn:
            return conn.execute("DELETE FROM v2_outbox WHERE audience_user_id=?", (user_id,)).rowcount

    def user_context(self, user_id: str) -> dict[str, str]:
        return {"user_id": user_id, "content": self.runtime.user_store.read_user_context(user_id)}

    def save_user_context(self, user_id: str, content: str) -> dict[str, str]:
        return {"user_id": user_id, "path": self.runtime.user_store.write_user_context(user_id, content)}

    def bind_user_address(self, **values: Any) -> dict[str, object]:
        return self.runtime.user_store.bind_address(**values).to_dict()

    def remove_user_address(self, address_id: str) -> bool:
        return self.runtime.user_store.remove_address(address_id)

    def set_user_aliases(self, *, user_id: str, aliases: list[str]) -> dict[str, object]:
        return {"user_id": user_id, "aliases": self.runtime.user_store.set_aliases(user_id, aliases)}

    def pending_access_requests(self) -> list[dict[str, object]]:
        return [request.to_dict() for request in self.runtime.user_store.list_access_requests()]

    def approve_access_request(self, *, request_id: str, display_name: str = "", user_id: str = "") -> dict[str, object]:
        request, address = self.runtime.user_store.approve_access_request(
            request_id,
            display_name=display_name,
            user_id=user_id,
        )
        if request.provider_key == "telegram":
            record = self.runtime.integration_store.get(request.integration_id)
            if record is None:
                raise ValueError("access_request_integration_not_found")
            config = dict(record.config)
            allowed = {str(value).strip() for value in config.get("allowed_chat_ids", []) if str(value).strip()}
            allowed.add(request.channel_target)
            self.runtime.integration_store.upsert(
                integration_id=record.integration_id,
                provider_key=record.provider_key,
                display_name=record.display_name,
                enabled=record.enabled,
                config={**config, "allowed_chat_ids": sorted(allowed)},
                secrets=record.secrets,
            )
            refresh_runtime_identity_resolver(self.runtime)
            self.restart_integrations()
        return {"request": request.to_dict(), "address": address.to_dict()}

    def reject_access_request(self, *, request_id: str) -> dict[str, object]:
        return self.runtime.user_store.reject_access_request(request_id).to_dict()

    def settings(self) -> dict[str, object]:
        return self.runtime.user_store.status()

    def save_settings(self, *, users_root: str) -> dict[str, object]:
        return {"users_root": self.runtime.user_store.set_users_root(users_root), "warning_repository_path": "/Alphonse/" in str(users_root)}

    def memory_settings(self, *, actor_user_id: str) -> dict[str, object]:
        self._require_admin(actor_user_id)
        return self.runtime.memory_settings_store.get().to_dict()

    def save_memory_settings(self, *, actor_user_id: str, values: dict[str, Any]) -> dict[str, object]:
        self._require_admin(actor_user_id)
        current = self.runtime.memory_settings_store.get()
        saved = self.runtime.memory_settings_store.save(MemorySettings(
            max_ledger_bytes=values.get("max_ledger_bytes", current.max_ledger_bytes),
            compaction_summary_max_words=values.get("compaction_summary_max_words", current.compaction_summary_max_words),
        ))
        return saved.to_dict()

    def web_tools_settings(self, *, actor_user_id: str) -> dict[str, object]:
        self._require_admin(actor_user_id)
        return self.runtime.web_tools_settings_store.get().to_dict()

    def save_web_tools_settings(self, *, actor_user_id: str, values: dict[str, Any]) -> dict[str, object]:
        self._require_admin(actor_user_id)
        current = self.runtime.web_tools_settings_store.get()
        saved = self.runtime.web_tools_settings_store.save(WebToolsSettings(
            enabled=bool(values.get("enabled", current.enabled)), searxng_base_url=str(values.get("searxng_base_url", current.searxng_base_url)),
            search_timeout_seconds=values.get("search_timeout_seconds", current.search_timeout_seconds), fetch_timeout_seconds=values.get("fetch_timeout_seconds", current.fetch_timeout_seconds),
            fetch_max_chars=values.get("fetch_max_chars", current.fetch_max_chars),
        ))
        refresh_runtime_web_tools(self.runtime)
        return saved.to_dict()

    def verify_web_tools(self, *, actor_user_id: str, kind: str) -> dict[str, Any]:
        self._require_admin(actor_user_id)
        settings = self.runtime.web_tools_settings_store.get()
        if kind == "search": return execute_web_search({"query": "Alphonse SearXNG verification", "limit": 1}, settings=settings)
        if kind == "fetch": return execute_web_fetch({"url": "https://example.com", "max_chars": 200}, settings=settings)
        raise ValueError("web_tools_verify_kind_invalid")

    def media_tools_settings(self, *, actor_user_id: str) -> dict[str, object]:
        self._require_admin(actor_user_id)
        return self.runtime.media_tools_settings_store.get().to_dict()

    def save_media_tools_settings(self, *, actor_user_id: str, kind: str, values: dict[str, Any]) -> dict[str, object]:
        self._require_admin(actor_user_id)
        saved = self.runtime.media_tools_settings_store.update(kind, values)
        refresh_runtime_media_tools(self.runtime)
        return saved.to_dict()

    def verify_media_tools(self, *, actor_user_id: str, kind: str, sample: str = "") -> dict[str, Any]:
        self._require_admin(actor_user_id)
        settings = self.runtime.media_tools_settings_store.get()
        if kind == "tts": result = verify_tts(settings.tts, sample_text=sample or "Alphonse text-to-speech verification.")
        elif kind == "stt": result = verify_stt(settings.stt, sample_path=sample)
        elif kind == "ocr": result = verify_ocr(settings.ocr, sample_path=sample)
        else: raise ValueError("media_tools_kind_invalid")
        exception = result.get("exception") if isinstance(result, dict) else {"message": "verification_failed"}
        output = result.get("output") if isinstance(result, dict) else {}
        preview = str((output or {}).get("text") or (output or {}).get("file_path") or "")
        error = str((exception or {}).get("message") or "") if isinstance(exception, dict) else ""
        details = (exception or {}).get("details") if isinstance(exception, dict) else {}
        detail_error = str((details or {}).get("error") or "") if isinstance(details, dict) else ""
        if detail_error:
            error = f"{error}: {detail_error}"
        saved = self.runtime.media_tools_settings_store.mark_verification(kind, ready=not bool(exception), error=error, preview=preview)
        refresh_runtime_media_tools(self.runtime)
        return {"result": result, "settings": saved.to_dict()}

    def _require_admin(self, actor_user_id: str) -> None:
        actor = str(actor_user_id or "").strip()
        if not actor or not self.runtime.user_store.is_admin(actor):
            raise PermissionError("admin_required")

    def list_agent_config(self) -> list[dict[str, str]]:
        return [document.to_dict(include_content=False) for document in self.runtime.agent_config_store.list_documents()]

    def read_agent_config(self, file_name: str) -> dict[str, str]:
        return self.runtime.agent_config_store.read(file_name).to_dict()

    def save_agent_config(self, file_name: str, content: str) -> dict[str, str]:
        return self.runtime.agent_config_store.save(file_name, content).to_dict()

    def pop_activity_events(self) -> list[dict[str, Any]]:
        """Legacy destructive-looking API backed by a shared event journal.

        Old TUI clients retain their one-shot polling behaviour while Desktop
        clients can use a cursor without stealing events from the TUI.
        """
        with self._event_lock:
            self._collect_activity_events()
            events = [event for event in self._activity_event_journal if int(event["sequence"]) > self._legacy_activity_cursor]
            if events:
                self._legacy_activity_cursor = int(events[-1]["sequence"])
            return [_without_sequence(event) for event in events]

    def activity_events_since(
        self,
        *,
        after_sequence: int = 0,
        integration_id: str = "",
        channel_target: str = "",
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return a non-destructive, client-cursored view of activity."""
        with self._event_lock:
            self._collect_activity_events()
            cursor = max(0, int(after_sequence or 0))
            matched = [
                event
                for event in self._activity_event_journal
                if int(event["sequence"]) > cursor
                and (not integration_id or event["integration_id"] == integration_id)
                and (not channel_target or event["channel_target"] == channel_target)
            ][: max(1, min(int(limit or 100), 500))]
            next_sequence = int(matched[-1]["sequence"]) if matched else cursor
            return matched, next_sequence

    def poll_desktop(
        self,
        *,
        client_id: str,
        user: str,
        after_sequence: int = 0,
        after_ui_sequence: int = 0,
        client_capabilities: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Collect Desktop activity and atomically lease its pending messages."""
        normalized_user = self._admin_user_id(user)
        normalized_client = str(client_id or "desktop").strip() or "desktop"
        capabilities = dict(client_capabilities or {})
        catalogs = capabilities.get("supportedCatalogIds")
        self._desktop_capabilities[normalized_client] = {
            str(catalog).strip() for catalog in catalogs if str(catalog).strip()
        } if isinstance(catalogs, list) else set()
        events, next_sequence = self.activity_events_since(
            after_sequence=after_sequence,
            integration_id="desktop",
            channel_target=normalized_user,
            limit=limit,
        )
        selector = OutboundSelector(integration_id="desktop", channel_target=normalized_user, status="pending")
        deliveries: list[dict[str, Any]] = []
        for _ in range(max(1, min(int(limit or 20), 100))):
            delivery = self.runtime.outbox.claim_next(selector, lease_owner=f"desktop:{normalized_client}", lease_seconds=120)
            if delivery is None:
                break
            deliveries.append(delivery.to_dict())
        questions = [question.to_dict() for question in self.runtime.question_store.list_pending_for_respondent(normalized_user)]
        ui_events, next_ui_sequence = self.ui_events_since(
            after_sequence=after_ui_sequence,
            user=normalized_user,
            limit=limit,
        )
        if ALPHONSE_DESKTOP_CATALOG_ID in self._desktop_capabilities[normalized_client]:
            ui_events = self._a2ui_scheduled_task_events(ui_events)
            ui_events.extend(self._sync_question_surfaces(client_id=normalized_client, user=normalized_user))
        else:
            ui_events = [item for item in ui_events if _event_name(item) != "scheduled_task_created"]
        return {
            "events": events,
            "next_sequence": next_sequence,
            "deliveries": deliveries,
            "questions": questions,
            "ui_events": ui_events,
            "next_ui_sequence": next_ui_sequence,
            "server_capabilities": self._a2ui.server_capabilities(),
            "status": {"active_work": self.active_work(), "activity": self.activity_status()},
        }

    def acknowledge_desktop_delivery(self, *, client_id: str, outbox_message_id: str) -> bool:
        delivery = self.runtime.outbox.get(outbox_message_id)
        expected_owner = f"desktop:{str(client_id or 'desktop').strip() or 'desktop'}"
        if delivery is None or delivery.integration_id != "desktop" or delivery.lease_owner != expected_owner:
            return False
        return self.runtime.outbox.mark_delivered(outbox_message_id)

    def desktop_conversation_history(self, *, user: str, project_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        normalized_user = self._admin_user_id(user)
        normalized_project = str(project_id or "").strip()
        timeline = self.runtime.conversation_store.list(owner_user_id=normalized_user, project_id=normalized_project, limit=limit)
        if timeline:
            return [{"id": event.event_id, "role": event.role, "content": event.content, "source": event.source, "created_at": event.created_at} for event in timeline]
        if not normalized_project:
            return []
        legacy = self.runtime.core.memory.latest_content(user_id=normalized_user, project_id=normalized_project)
        recovered = legacy_ledger_events(legacy, owner_user_id=normalized_user, project_id=normalized_project, limit=limit)
        if recovered:
            return recovered
        deliveries = self.runtime.outbox.list(OutboundSelector(status=None), limit=max(1, min(int(limit or 100), 500)))
        return [
            {"id": delivery.outbox_message_id, "role": "assistant", "content": delivery.message, "created_at": delivery.created_at}
            for delivery in deliveries
            if delivery.audience_user_id == normalized_user and str(delivery.metadata.get("project_id") or "").strip() == normalized_project
        ][-max(1, min(int(limit or 100), 500)):]

    def list_projects(self, *, user: str) -> list[dict[str, str]]:
        normalized = self._admin_user_id(user)
        return [project.to_dict() for project in self.runtime.project_store.list_visible_projects(normalized, requester_is_admin=True)]

    def project_recent_files(self, *, user: str, project_id: str, limit: int = 4) -> list[dict[str, str]]:
        """Return the newest accessible direct children of an authorized project root."""
        actor = self._admin_user_id(user)
        project = self.runtime.project_store.get_project(
            project_id,
            requester_user_id=actor,
            requester_is_admin=True,
        )
        if project is None:
            raise ValueError("project_not_found")

        root = Path(project.root_path).expanduser()
        if not root.is_dir():
            return []
        entries: list[tuple[float, dict[str, str]]] = []
        try:
            children = root.iterdir()
            for child in children:
                if child.name.startswith(".") or not os.access(child, os.R_OK):
                    continue
                try:
                    stat = child.stat()
                    kind = "directory" if child.is_dir() else "file"
                except OSError:
                    continue
                entries.append((stat.st_mtime, {
                    "name": child.name,
                    "kind": kind,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }))
        except OSError:
            return []
        entries.sort(key=lambda item: item[0], reverse=True)
        bounded_limit = max(1, min(int(limit or 4), 4))
        return [entry for _, entry in entries[:bounded_limit]]

    def manageable_projects(self, *, user: str, status: str = "") -> list[dict[str, Any]]:
        actor = self._admin_user_id(user)
        status_value = str(status or "").strip() or None
        projects = self.runtime.project_store.list_manageable_projects(actor, requester_is_admin=True, status=status_value)  # type: ignore[arg-type]
        users = {item["user_id"]: item for item in self.list_users()}
        return [{**project.to_dict(), "owner": users.get(project.owner_user_id)} for project in projects]

    def create_project(self, *, user: str, name: str, description: str, root_path: str, visibility: str) -> dict[str, str]:
        owner = self._admin_user_id(user)
        from pathlib import Path
        slug = "-".join(part for part in "".join(char.lower() if char.isalnum() else " " for char in name).split()) or "project"
        parent = Path(root_path).expanduser() if str(root_path).strip() else self.runtime.user_store.managed_project_root(owner)
        project = self.runtime.project_store.create_project(
            name=name,
            description=description,
            root_path=str(parent / slug),
            visibility=visibility,  # type: ignore[arg-type]
            owner_user_id=owner,
        )
        return project.to_dict()

    def import_project(self, *, user: str, name: str, description: str, root_path: str, visibility: str) -> dict[str, Any]:
        owner = self._admin_user_id(user)
        root = Path(root_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError("project_import_directory_required")
        if self.runtime.project_store.find_project_by_root(str(root)) is not None:
            raise ValueError("project_root_already_registered")
        return self.runtime.project_store.create_project(name=name, description=description, root_path=str(root), visibility=visibility, owner_user_id=owner).to_dict()  # type: ignore[arg-type]

    def update_project(self, *, user: str, project_id: str, name: str, description: str, visibility: str) -> dict[str, Any]:
        actor = self._admin_user_id(user)
        return self.runtime.project_store.update_project(project_id, name=name, description=description, visibility=visibility, requester_user_id=actor, requester_is_admin=True).to_dict()  # type: ignore[arg-type]

    def archive_project(self, *, user: str, project_id: str) -> dict[str, Any]:
        actor = self._admin_user_id(user)
        self._ensure_project_has_no_live_schedules(project_id)
        project = self.runtime.project_store.archive_project(project_id, requester_user_id=actor, requester_is_admin=True)
        self.runtime.project_session_store.clear_project(project.project_id)
        if self.runtime.active_project_id == project.project_id: self.runtime.active_project_id = ""
        return project.to_dict()

    def restore_project(self, *, user: str, project_id: str) -> dict[str, Any]:
        actor = self._admin_user_id(user)
        return self.runtime.project_store.restore_project(project_id, requester_user_id=actor, requester_is_admin=True).to_dict()

    def delete_project(self, *, user: str, project_id: str, confirmation: str) -> dict[str, Any]:
        actor = self._admin_user_id(user)
        if str(confirmation or "") != str(project_id or ""):
            raise ValueError("delete_confirmation_must_match_project_id")
        self._ensure_project_has_no_live_schedules(project_id)
        project = self.runtime.project_store.delete_project(project_id, requester_user_id=actor, requester_is_admin=True)
        self.runtime.project_session_store.clear_project(project.project_id)
        if self.runtime.active_project_id == project.project_id: self.runtime.active_project_id = ""
        root = Path(project.root_path).resolve()
        managed = self.runtime.user_store.managed_project_root(project.owner_user_id).resolve()
        removed_files = False
        try:
            root.relative_to(managed)
            if root.exists():
                shutil.rmtree(root)
                removed_files = True
        except ValueError:
            pass
        return {"deleted": True, "project_id": project.project_id, "removed_managed_files": removed_files}

    def _ensure_project_has_no_live_schedules(self, project_id: str) -> None:
        tasks = self.runtime.schedule_store.list_tasks(project_id=str(project_id or ""), limit=1000)
        if any(task.status in {"active", "paused"} for task in tasks):
            raise ValueError("project_has_active_scheduled_tasks")

    def project_members(self, project_id: str) -> list[str]:
        self._admin_user_id()
        return self.runtime.project_store.list_members(project_id)

    def add_project_member(self, project_id: str, user_id: str) -> None:
        self._admin_user_id()
        if self.runtime.user_store.get_user(user_id) is None:
            raise KeyError("user_not_found")
        self.runtime.project_store.add_member(project_id, user_id)

    def remove_project_member(self, project_id: str, user_id: str) -> bool:
        self._admin_user_id()
        return self.runtime.project_store.remove_member(project_id, user_id)

    def select_project_session(
        self,
        *,
        user: str,
        integration_id: str,
        channel_target: str,
        thread_id: str,
        project_id: str,
    ) -> dict[str, str]:
        user = self._admin_user_id(user)
        key = ProjectSessionKey(user, integration_id, channel_target or user, thread_id)
        project = self.runtime.inbound_router.select_project(key, project_id)
        return self.runtime.project_session_store.set(key, project).to_dict()

    def active_project_session(self, *, user: str, integration_id: str, channel_target: str, thread_id: str) -> dict[str, str] | None:
        user = self._admin_user_id(user)
        key = ProjectSessionKey(user, integration_id, channel_target or user, thread_id)
        project = self.runtime.inbound_router.active_project(key)
        if project is None:
            return None
        session = self.runtime.project_session_store.get(key)
        return session.to_dict() if session is not None else None

    def ingest_message(self, **values: Any) -> dict[str, Any]:
        routed = self.runtime.inbound_router.ingest(**values)
        return {
            "message_id": routed.queued.message_id if routed.queued is not None else "",
            "handled_command": routed.handled_command,
            "project_id": routed.project_id,
        }

    def read_project_context(self, *, user: str, project_id: str) -> dict[str, str]:
        user = self._admin_user_id(user)
        project = self.runtime.project_store.get_project(project_id, requester_user_id=user, requester_is_admin=True)
        if project is None:
            raise KeyError(f"project_not_found: {project_id}")
        return {"project_id": project.project_id, "content": self.runtime.project_store.read_project_context(project.project_id, requester_user_id=user)}

    def save_project_context(self, *, user: str, project_id: str, content: str) -> dict[str, str]:
        return self.runtime.project_store.write_project_context(project_id, content, requester_user_id=self._admin_user_id(user), requester_is_admin=True).to_dict()

    def answer_question(self, *, user: str, question_id: str, text: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_user = self._admin_user_id(user)
        result = self.runtime.question_store.route_answer(
            respondent_user_id=normalized_user,
            question_id=question_id,
            text=text or None,
            payload=payload,
        )
        if result.handled and result.question is not None:
            child_id = str(result.question.metadata.get("child_task_id") or "").strip()
            if child_id:
                child = self.runtime.question_store.load_task_checkpoint(child_id)
                if child is not None:
                    self.runtime.core.memory.event(child, "Conversation", f"- {normalized_user}: {text or _question_answer_text(payload)}")
                    child.status = "completed"
                    child.outcome = {"status": "success", "answered_question_id": result.question.question_id}
                    self.runtime.core.memory.finish_task(child)
        if result.handled and result.resumed_task is not None:
            self.runtime.ui_events.append(
                CoreUiEvent(
                    event_type="question_interrupt_resolved",
                    payload={"question": result.question.to_dict() if result.question else None, "answer": result.answer},
                )
            )
            queued = self.runtime.channel.queue_message(
                prompt=text or _question_answer_text(payload),
                user=normalized_user,
                project_id=result.resumed_task.project_id,
                correlation_id=result.resumed_task.correlation_id,
                metadata={"task_state": result.resumed_task.to_dict(), "answered_question_id": result.question.question_id if result.question else ""},
                integration_id="desktop",
                provider_key="tui",
                channel_target=normalized_user,
            )
            payload_result = result.to_dict()
            payload_result["message_id"] = queued.message_id
            return payload_result
        return result.to_dict()

    def cancel_question(self, question_id: str) -> bool:
        question = self.runtime.question_store.get_question(question_id)
        cancelled = self.runtime.question_store.cancel_question(question_id)
        if cancelled:
            self.runtime.ui_events.append(
                CoreUiEvent(
                    event_type="question_interrupt_cancelled",
                    payload={"question": question.to_dict() if question else None, "cancelled": True},
                )
            )
        return cancelled

    def a2ui_action(
        self,
        *,
        client_id: str,
        user: str,
        surface_id: str,
        source_component_id: str,
        action_name: str,
        context: dict[str, Any] | None = None,
        data_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform a validated action from a server-owned A2UI surface."""
        client = str(client_id or "desktop").strip() or "desktop"
        respondent = self._admin_user_id(user)
        if ALPHONSE_DESKTOP_CATALOG_ID not in self._desktop_capabilities.get(client, set()):
            raise ValueError("a2ui_catalog_not_negotiated")
        values = dict(context or {})
        action = str(action_name or "").strip()
        source = str(source_component_id or "").strip()
        scheduled_task_id = _scheduled_task_id_from_surface(surface_id)
        if action == "view_scheduled_task":
            if source != "view" or not scheduled_task_id or scheduled_task_id != str(values.get("scheduled_task_id") or ""):
                raise ValueError("a2ui_surface_or_context_invalid")
            self._scheduled_task_for_actor(actor_user_id=respondent, scheduled_task_id=scheduled_task_id)
            return {"action": "view_scheduled_task", "scheduled_task_id": scheduled_task_id}
        question_id = question_id_from_surface(surface_id)
        if not question_id or question_id != str(values.get("question_id") or ""):
            raise ValueError("a2ui_surface_or_context_invalid")
        question = self.runtime.question_store.get_question(question_id)
        if question is None or question.status != "pending" or question.respondent_user_id != respondent:
            raise ValueError("a2ui_question_not_available")
        if action == "cancel_question" and source == "cancel":
            return {"cancelled": self.cancel_question(question_id)}
        if action != "answer_question":
            raise ValueError("a2ui_action_not_allowed")
        payload: dict[str, Any]
        text = ""
        if question.kind == "yes_no":
            answer = values.get("answer")
            if source not in {"answer_yes", "answer_no"} or not isinstance(answer, bool):
                raise ValueError("a2ui_answer_invalid")
            if (source == "answer_yes") != answer:
                raise ValueError("a2ui_source_invalid")
            payload, text = {"answer": answer}, "yes" if answer else "no"
        elif question.kind == "single_choice":
            choice_id = str(values.get("choice_id") or "")
            if source != f"choice_{choice_id}" or choice_id not in {choice.id for choice in question.choices}:
                raise ValueError("a2ui_choice_invalid")
            label = next(choice.label for choice in question.choices if choice.id == choice_id)
            payload, text = {"choice_id": choice_id}, label
        else:
            model = dict(data_model or {})
            answer = model.get("answer") if isinstance(model.get("answer"), dict) else {}
            text = str(answer.get("text") or "").strip()
            if source != "submit" or not text:
                raise ValueError("a2ui_text_invalid")
            payload = {"text": text}
        return self.answer_question(user=respondent, question_id=question_id, text=text, payload=payload)

    def _a2ui_scheduled_task_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for item in events:
            event = item.get("event") if isinstance(item.get("event"), dict) else {}
            if event.get("type") != "CUSTOM" or event.get("name") != "scheduled_task_created":
                rendered.append(item)
                continue
            payload = event.get("value") if isinstance(event.get("value"), dict) else {}
            task = payload.get("scheduled_task") if isinstance(payload.get("scheduled_task"), dict) else {}
            try:
                rendered.extend({"event": _a2ui_custom(envelope)} for envelope in self._a2ui.scheduled_task_created(task, project_name=str(payload.get("project_name") or "")))
            except ValueError:
                continue
        return rendered

    def list_integrations(self) -> list[dict[str, Any]]:
        records = {record.provider_key: record for record in self.runtime.integration_store.list()}
        return [
            {
                "provider_key": descriptor.provider_key,
                "display_name": descriptor.display_name,
                "integration": records[descriptor.provider_key].to_dict() if descriptor.provider_key in records else None,
            }
            for descriptor in self.runtime.integration_registry.list()
        ]

    def save_telegram_integration(self, *, user: str, values: dict[str, Any]) -> dict[str, Any]:
        existing = self.runtime.integration_store.get(str(values.get("integration_id") or "")) or self.runtime.integration_store.get_by_provider("telegram")
        secrets = dict(existing.secrets) if existing is not None else {}
        token = str(values.get("bot_token") or "").strip()
        if bool(values.get("remove_token")):
            secrets.pop("bot_token", None)
        elif token:
            secrets["bot_token"] = token
        enabled = bool(values.get("enabled"))
        if enabled and not str(secrets.get("bot_token") or "").strip():
            raise ValueError("telegram_bot_token_required")
        provider_user_id = str(values.get("telegram_user_id") or "").strip()
        if provider_user_id:
            self.runtime.user_store.bind_address(
                user_id=self._admin_user_id(user),
                integration_id=str(values.get("integration_id") or "telegram-home"),
                provider_key="telegram",
                provider_user_id=provider_user_id,
            )
        record = self.runtime.integration_store.upsert(
            integration_id=str(values.get("integration_id") or "telegram-home").strip() or "telegram-home",
            provider_key="telegram",
            display_name=str(values.get("display_name") or "Telegram").strip() or "Telegram",
            enabled=enabled,
            config={
                "poll_interval_sec": _positive_float(values.get("poll_interval_sec")),
                "allowed_chat_ids": _comma_values(values.get("allowed_chat_ids")),
                "owner_user_id": self._admin_user_id(user),
                "telegram_user_id": provider_user_id,
                "presence_enabled": bool(values.get("presence_enabled", True)),
            },
            secrets=secrets,
        )
        refresh_runtime_identity_resolver(self.runtime)
        self.restart_integrations()
        return record.to_dict()

    def save_discord_integration(self, *, user: str, values: dict[str, Any]) -> dict[str, Any]:
        existing = self.runtime.integration_store.get(str(values.get("integration_id") or "")) or self.runtime.integration_store.get_by_provider("discord")
        secrets = dict(existing.secrets) if existing is not None else {}
        token = str(values.get("bot_token") or "").strip()
        if bool(values.get("remove_token")):
            secrets.pop("bot_token", None)
        elif token:
            secrets["bot_token"] = token
        enabled = bool(values.get("enabled"))
        if enabled and not str(secrets.get("bot_token") or "").strip():
            raise ValueError("discord_bot_token_required")
        integration_id = str(values.get("integration_id") or "discord-home").strip() or "discord-home"
        provider_user_id = str(values.get("discord_user_id") or "").strip()
        if provider_user_id:
            self.runtime.user_store.bind_address(
                user_id=self._admin_user_id(user), integration_id=integration_id,
                provider_key="discord", provider_user_id=provider_user_id,
            )
        record = self.runtime.integration_store.upsert(
            integration_id=integration_id,
            provider_key="discord",
            display_name=str(values.get("display_name") or "Discord").strip() or "Discord",
            enabled=enabled,
            config={
                "allowed_guild_ids": _comma_values(values.get("allowed_guild_ids")),
                "allowed_channel_ids": _comma_values(values.get("allowed_channel_ids")),
                "owner_user_id": self._admin_user_id(user),
                "discord_user_id": provider_user_id,
                "presence_enabled": bool(values.get("presence_enabled", True)),
            },
            secrets=secrets,
        )
        refresh_runtime_identity_resolver(self.runtime)
        self.restart_integrations()
        return record.to_dict()

    def _collect_activity_events(self) -> None:
        events = list(self.runtime.activity_events)
        self.runtime.activity_events.clear()
        for event in events:
            self._activity_event_sequence += 1
            self._activity_event_journal.append(
                {
                    "sequence": self._activity_event_sequence,
                    "phase": event.phase.value,
                    "label": event.label,
                    "message": event.message,
                    "speaker": event.speaker,
                    "task_id": event.task_id,
                    "message_id": event.message_id,
                    "user": event.user,
                    "integration_id": event.integration_id,
                    "channel_target": event.channel_target,
                }
            )
        if len(self._activity_event_journal) > 2000:
            self._activity_event_journal = self._activity_event_journal[-2000:]

    def ui_events_since(self, *, after_sequence: int = 0, user: str, limit: int = 100) -> tuple[list[dict[str, Any]], int]:
        """Return ordered AG-UI events without changing TUI activity polling."""
        with self._ui_event_lock:
            self._collect_ui_events()
            cursor = max(0, int(after_sequence or 0))
            matched = [
                event
                for event in self._ui_event_journal
                if int(event["sequence"]) > cursor and (not event["user"] or event["user"] == user)
            ][: max(1, min(int(limit or 100), 500))]
            return matched, int(matched[-1]["sequence"]) if matched else cursor

    def _collect_ui_events(self) -> None:
        events = list(self.runtime.ui_events)
        self.runtime.ui_events.clear()
        for core_event in events:
            for event in self._ag_ui.map_event(core_event):
                self._ui_event_sequence += 1
                self._ui_event_journal.append(
                    {
                        "sequence": self._ui_event_sequence,
                        "user": _ui_event_user(core_event),
                        "event": event,
                    }
                )
        if len(self._ui_event_journal) > 2000:
            self._ui_event_journal = self._ui_event_journal[-2000:]

    def _sync_question_surfaces(self, *, client_id: str, user: str) -> list[dict[str, Any]]:
        """Reconcile trusted question surfaces per Desktop client.

        This is deliberately state based: a reconnect (and question expiry) is
        recoverable even if a prior poll response was lost.
        """
        key = (client_id, user)
        known = self._desktop_surfaces.setdefault(key, set())
        pending = self.runtime.question_store.list_pending_for_respondent(user)
        expected = {surface_id_for_question(question.question_id): question for question in pending}
        events: list[dict[str, Any]] = []
        for surface_id in sorted(known - set(expected)):
            events.append({"event": _a2ui_custom(self._a2ui.question_closed(question_id_from_surface(surface_id)))})
        for surface_id, question in expected.items():
            if surface_id not in known:
                events.extend({"event": _a2ui_custom(message)} for message in self._a2ui.question_opened(question))
        self._desktop_surfaces[key] = set(expected)
        return events

    def active_work(self) -> dict[str, str]:
        with self._active_work_lock:
            return dict(self._active_work)

    def activity_status(self) -> dict[str, str]:
        with self._active_work_lock:
            return dict(self._activity_status)

    def run_once(self) -> Any:
        queued = self.runtime.queue.peek()
        self._set_active_work(queued)
        if queued is not None:
            self._set_activity_status("working")
        with self.runtime.presence_projector.processing(queued):
            step = self.runtime.core.step()
            if step.status in {
                LoopStepStatus.PROCESSED,
                LoopStepStatus.PARKED,
                LoopStepStatus.WAITING,
                LoopStepStatus.FAILED,
            }:
                self.runtime.presence_projector.finish(
                    failed=step.status == LoopStepStatus.FAILED,
                    waiting=step.status in {LoopStepStatus.PARKED, LoopStepStatus.WAITING},
                )
        snapshot = self.runtime.visible_state.snapshot()
        if step.status in {LoopStepStatus.PROCESSED, LoopStepStatus.PARKED, LoopStepStatus.WAITING}:
            projected = project_snapshot_to_outbox(snapshot=snapshot, outbox=self.runtime.outbox)
            if projected is not None:
                self.runtime.conversation_store.record(owner_user_id=projected.audience_user_id, project_id=str(projected.metadata.get("project_id") or ""), role="assistant", content=projected.message, source=projected.integration_id, source_message_id=f"outbound:{projected.outbox_message_id}", created_at=projected.created_at)
                occurrence_key = str(projected.metadata.get("occurrence_key") or "").strip()
                if occurrence_key:
                    self.runtime.schedule_store.mark_occurrence_response_pending(
                        occurrence_key,
                        response_outbox_id=projected.outbox_message_id,
                    )
            self._emit_scheduled_task_card(snapshot)
            if step.queued_message_id:
                acknowledge = getattr(self.runtime.queue, "ack", None)
                if callable(acknowledge):
                    acknowledge(step.queued_message_id, lease_owner=self.daemon_id)
        elif step.status == LoopStepStatus.FAILED and step.queued_message_id:
            retry = getattr(self.runtime.queue, "retry", None)
            if callable(retry):
                retry(
                    step.queued_message_id,
                    error=str(getattr(step, "error", "") or "capd_processing_failed"),
                    next_attempt_at=datetime.now(timezone.utc) + timedelta(seconds=5),
                    lease_owner=self.daemon_id,
                    max_attempts=self.inbound_max_attempts,
                )
            self.runtime.core.clear_failure()
        if step.status != LoopStepStatus.BUSY:
            self._set_active_work(None)
            terminal_state = {
                LoopStepStatus.WAITING: "waiting",
                LoopStepStatus.PARKED: "waiting",
                LoopStepStatus.FAILED: "error",
            }.get(step.status, "idle")
            self._set_activity_status(terminal_state)
        return step

    def _set_activity_status(self, state: str) -> None:
        with self._active_work_lock:
            self._activity_status = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat()}

    def _set_active_work(self, queued: Any | None) -> None:
        with self._active_work_lock:
            if queued is None:
                self._active_work = {}
                return
            message = getattr(queued, "message", None)
            self._active_work = {
                "message_id": str(getattr(queued, "message_id", "") or ""),
                "user": str(getattr(message, "user", "") or ""),
                "prompt": str(getattr(message, "prompt", "") or ""),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.wait(0.5):
                pass
        finally:
            self.stop()

    def _emit_scheduled_task_card(self, snapshot: Any) -> None:
        metadata = getattr(snapshot, "metadata", {}) or {}
        task_state = metadata.get("task_state") if isinstance(metadata, dict) else None
        result = _scheduled_task_result(task_state)
        if result is None:
            return
        task_metadata = task_state.get("metadata") if isinstance(task_state.get("metadata"), dict) else {}
        channel = task_metadata.get("channel") if isinstance(task_metadata.get("channel"), dict) else {}
        if str(channel.get("integration_id") or "") != "desktop":
            return
        record = self.runtime.schedule_store.get_task(str(result.get("scheduled_task_id") or ""))
        if record is None:
            return
        project_id = str(task_state.get("project_id") or "").strip()
        project = self.runtime.project_store.get_project(project_id, requester_user_id=str(task_state.get("user") or ""), requester_is_admin=True) if project_id else None
        self.runtime.ui_events.append(
            CoreUiEvent(
                event_type="scheduled_task_created",
                payload={
                    "task_state": task_state,
                    "scheduled_task": {**record.to_dict(), "schedule_summary": schedule_summary(record.schedule)},
                    "project_name": project.name if project is not None else "",
                },
            )
        )

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                step = self.run_once()
                self._last_processor_error = ""
            except Exception as exc:
                self._last_processor_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                self._stop.wait(max(0.1, self.poll_interval_sec))
                continue
            if step.status in {LoopStepStatus.EMPTY, LoopStepStatus.BUSY}:
                self._stop.wait(max(0.01, self.poll_interval_sec))

    def _acquire_single_instance_lock(self) -> None:
        lock_path = default_socket_path().with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = lock_path.open("a+")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                V2DaemonClient(default_socket_path(), timeout_sec=0.5).ping()
            except Exception as exc:
                raise RuntimeError("alphonse_v2_daemon_lock_held") from exc
            raise RuntimeError("alphonse_v2_daemon_already_running")
        self._lock_file.seek(0)
        self._lock_file.truncate()
        self._lock_file.write(self.daemon_id)
        self._lock_file.flush()

    def _release_single_instance_lock(self) -> None:
        with self._lifecycle_lock:
            lock_file = self._lock_file
            self._lock_file = None
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        finally:
            pass

    def _on_outbox_delivered(self, outbound: Any) -> None:
        message_id = str(getattr(outbound, "outbox_message_id", "") or "")
        if message_id:
            delivered = self.runtime.outbox.get(message_id)
            self.runtime.communication_router.threads.mark_delivered(
                message_id,
                str(getattr(delivered, "provider_message_id", "") or ""),
            )
        metadata = getattr(outbound, "metadata", {}) if outbound is not None else {}
        occurrence_key = str(metadata.get("occurrence_key") or "").strip() if isinstance(metadata, dict) else ""
        if occurrence_key:
            self.runtime.schedule_store.mark_occurrence_delivered(
                occurrence_key,
                response_outbox_id=str(getattr(outbound, "outbox_message_id", "") or ""),
            )

    def _on_outbox_failed(self, outbound: Any, error: str) -> None:
        message_id = str(getattr(outbound, "outbox_message_id", "") or "")
        if message_id:
            thread = self.runtime.communication_router.threads.mark_failed(message_id)
            if thread is not None:
                self.runtime.outbox.enqueue(
                    address=thread.origin,
                    message="I could not deliver your message. Please try again later.",
                    kind="communication_delivery_failed",
                    audience_user_id=thread.sender_user_id,
                    metadata={"communication_thread_id": thread.thread_id},
                )
        metadata = getattr(outbound, "metadata", {}) if outbound is not None else {}
        occurrence_key = str(metadata.get("occurrence_key") or "").strip() if isinstance(metadata, dict) else ""
        if occurrence_key:
            self.runtime.schedule_store.mark_occurrence_failed(occurrence_key, error=error)


def _without_sequence(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "sequence"}


def _ui_event_user(event: CoreUiEvent) -> str:
    payload = dict(event.payload or {})
    question = payload.get("question")
    if isinstance(question, dict):
        return str(question.get("respondent_user_id") or "").strip()
    task = payload.get("task") or payload.get("task_state")
    if isinstance(task, dict):
        return str(task.get("user") or "").strip()
    return str(payload.get("user") or "").strip()


def _a2ui_custom(envelope: dict[str, Any]) -> dict[str, Any]:
    return {"type": "CUSTOM", "name": "a2ui.envelope", "value": envelope}


def _event_name(item: dict[str, Any]) -> str:
    event = item.get("event") if isinstance(item.get("event"), dict) else {}
    return str(event.get("name") or "")


def _scheduled_task_id_from_surface(surface_id: str) -> str:
    value = str(surface_id or "").strip()
    return value.removeprefix("scheduled-task:") if value.startswith("scheduled-task:") else ""


def _scheduled_task_result(task_state: Any) -> dict[str, Any] | None:
    if not isinstance(task_state, dict):
        return None
    calls = task_state.get("plan_json")
    try:
        import json
        calls = json.loads(calls) if isinstance(calls, str) else calls
    except (TypeError, ValueError):
        return None
    if not isinstance(calls, list):
        return None
    for call in reversed(calls):
        if not isinstance(call, dict) or str(call.get("tool_id") or "") != "native.scheduled_task":
            continue
        execution = call.get("execution") if isinstance(call.get("execution"), dict) else {}
        result = execution.get("result") if isinstance(execution.get("result"), dict) else {}
        if str(execution.get("status") or "") == "success" and str(result.get("scheduled_task_id") or ""):
            return result
    return None


def _comma_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return parsed if parsed > 0 else 1.0


def _question_answer_text(payload: dict[str, Any] | None) -> str:
    value = dict(payload or {})
    if "text" in value:
        return str(value["text"] or "")
    if "choice_id" in value:
        return str(value["choice_id"] or "")
    if "answer" in value:
        return "yes" if bool(value["answer"]) else "no"
    return "answer"


def main() -> None:
    daemon_id = f"daemon-{uuid4().hex[:12]}"
    user_store = V2UserStore.default()
    daemon = V2Daemon(
        build_runtime_host(
            user=user_store.admin_user().user_id if user_store.admin_user() else "",
            user_store=user_store,
            messages=SQLiteMessageQueue.default(lease_owner=daemon_id),
            question_store=SQLiteQuestionStore.default(),
            project_store=ProjectStore.default(),
            schedule_store=ScheduledTaskStore.default(),
            web_tools_settings_store=SQLiteWebToolsSettingsStore.default(),
            media_tools_settings_store=SQLiteMediaToolsSettingsStore.default(),
            asset_store=SQLiteAssetStore.default(),
            conversation_store=SQLiteConversationStore.default(),
            memory_settings_store=SQLiteMemorySettingsStore.default(),
            outbox=SQLiteOutboundStore.default(),
            integration_store=SQLiteIntegrationStore.default(),
            inference_settings_store=SQLiteInferenceSettingsStore.default(),
            agent_config_store=AgentConfigStore.default(),
            project_session_store=SQLiteProjectSessionStore.default(),
        )
    )
    previous = {}

    def _request_stop(signum: int, frame: Any) -> None:
        _ = signum, frame
        daemon.stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, _request_stop)
    try:
        daemon.run_forever()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
