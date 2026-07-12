"""Foreground v2 daemon host."""

from __future__ import annotations

import signal
import fcntl
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
        start_runtime_integrations(
            self.runtime,
            on_outbox_delivered=self._on_outbox_delivered,
            on_outbox_failed=self._on_outbox_failed,
        )

    def update_inference_settings(self, *, provider_key: str, model_id: str) -> dict[str, str]:
        settings = validate_and_save_inference_settings(
            self.runtime.inference_settings_store,
            provider_key=provider_key,
            model_id=model_id,
        )
        refresh_runtime_inference(self.runtime, settings)
        return settings.to_dict()

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
        normalized_user = str(user or "local").strip() or "local"
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
            ui_events.extend(self._sync_question_surfaces(client_id=normalized_client, user=normalized_user))
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

    def list_projects(self, *, user: str) -> list[dict[str, str]]:
        return [project.to_dict() for project in self.runtime.project_store.list_visible_projects(str(user or "local"))]

    def create_project(self, *, user: str, name: str, description: str, root_path: str, visibility: str) -> dict[str, str]:
        project = self.runtime.project_store.create_project(
            name=name,
            description=description,
            root_path=root_path,
            visibility=visibility if visibility in {"private", "shared"} else "private",
            owner_user_id=str(user or "local"),
        )
        return project.to_dict()

    def select_project_session(
        self,
        *,
        user: str,
        integration_id: str,
        channel_target: str,
        thread_id: str,
        project_id: str,
    ) -> dict[str, str]:
        key = ProjectSessionKey(user, integration_id, channel_target, thread_id)
        project = self.runtime.inbound_router.select_project(key, project_id)
        return self.runtime.project_session_store.set(key, project).to_dict()

    def active_project_session(self, *, user: str, integration_id: str, channel_target: str, thread_id: str) -> dict[str, str] | None:
        key = ProjectSessionKey(user, integration_id, channel_target, thread_id)
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
        project = self.runtime.project_store.get_project(project_id, requester_user_id=user)
        if project is None:
            raise KeyError(f"project_not_found: {project_id}")
        return {"project_id": project.project_id, "content": self.runtime.project_store.read_project_context(project.project_id, requester_user_id=user)}

    def save_project_context(self, *, user: str, project_id: str, content: str) -> dict[str, str]:
        return self.runtime.project_store.write_project_context(project_id, content, requester_user_id=user).to_dict()

    def answer_question(self, *, user: str, question_id: str, text: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_user = str(user or "local").strip() or "local"
        result = self.runtime.question_store.route_answer(
            respondent_user_id=normalized_user,
            question_id=question_id,
            text=text or None,
            payload=payload,
        )
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
        """Perform the sole permitted A2UI mutation: answer or cancel a question."""
        client = str(client_id or "desktop").strip() or "desktop"
        respondent = str(user or "local").strip() or "local"
        if ALPHONSE_DESKTOP_CATALOG_ID not in self._desktop_capabilities.get(client, set()):
            raise ValueError("a2ui_catalog_not_negotiated")
        question_id = question_id_from_surface(surface_id)
        values = dict(context or {})
        if not question_id or question_id != str(values.get("question_id") or ""):
            raise ValueError("a2ui_surface_or_context_invalid")
        question = self.runtime.question_store.get_question(question_id)
        if question is None or question.status != "pending" or question.respondent_user_id != respondent:
            raise ValueError("a2ui_question_not_available")
        source = str(source_component_id or "").strip()
        action = str(action_name or "").strip()
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
            upsert_provider_user_mapping(
                alphonse_user_id=str(user or "local"), provider_key="telegram", provider_user_id=provider_user_id,
                display_name=str(user or "local"), is_active=True,
            )
        record = self.runtime.integration_store.upsert(
            integration_id=str(values.get("integration_id") or "telegram-home").strip() or "telegram-home",
            provider_key="telegram",
            display_name=str(values.get("display_name") or "Telegram").strip() or "Telegram",
            enabled=enabled,
            config={
                "poll_interval_sec": _positive_float(values.get("poll_interval_sec")),
                "allowed_chat_ids": _comma_values(values.get("allowed_chat_ids")),
                "owner_user_id": str(user or "local"),
                "telegram_user_id": provider_user_id,
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
                occurrence_key = str(projected.metadata.get("occurrence_key") or "").strip()
                if occurrence_key:
                    self.runtime.schedule_store.mark_occurrence_response_pending(
                        occurrence_key,
                        response_outbox_id=projected.outbox_message_id,
                    )
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
        metadata = getattr(outbound, "metadata", {}) if outbound is not None else {}
        occurrence_key = str(metadata.get("occurrence_key") or "").strip() if isinstance(metadata, dict) else ""
        if occurrence_key:
            self.runtime.schedule_store.mark_occurrence_delivered(
                occurrence_key,
                response_outbox_id=str(getattr(outbound, "outbox_message_id", "") or ""),
            )

    def _on_outbox_failed(self, outbound: Any, error: str) -> None:
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
    daemon = V2Daemon(
        build_runtime_host(
            user="local",
            messages=SQLiteMessageQueue.default(lease_owner=daemon_id),
            question_store=SQLiteQuestionStore.default(),
            project_store=ProjectStore.default(),
            schedule_store=ScheduledTaskStore.default(),
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
