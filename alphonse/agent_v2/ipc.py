"""Local Unix-socket protocol for v2 daemon clients."""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphonse.agent_v2.retention import storage_metrics
from uuid import uuid4


PROTOCOL_VERSION = 1


def default_socket_path() -> Path:
    return Path(os.getenv("ALPHONSE_V2_SOCKET_PATH") or Path.home() / ".alphonse" / "v2-daemon.sock")


class V2DaemonClient:
    def __init__(self, socket_path: str | Path | None = None, *, timeout_sec: float = 2.0) -> None:
        self.socket_path = Path(socket_path) if socket_path is not None else default_socket_path()
        self.timeout_sec = max(0.1, float(timeout_sec))

    def request(self, method: str, **params: Any) -> dict[str, Any]:
        request_id = uuid4().hex
        payload = json.dumps(
            {"version": PROTOCOL_VERSION, "request_id": request_id, "method": str(method), "params": params},
            sort_keys=True,
        ) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_sec)
            connection.connect(str(self.socket_path))
            connection.sendall(payload.encode("utf-8"))
            data = _read_line(connection)
        response = json.loads(data)
        if not isinstance(response, dict):
            raise RuntimeError("daemon_invalid_response")
        if response.get("request_id") not in {None, request_id}:
            raise RuntimeError("daemon_request_id_mismatch")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "daemon_request_failed"))
        return dict(response.get("result") or {})

    def ping(self) -> dict[str, Any]:
        return self.request("ping")

    def status(self) -> dict[str, Any]:
        return self.request("status")

    def runtime_identity(self) -> dict[str, Any]:
        return self.request("runtime_identity")

    def stop(self) -> dict[str, Any]:
        return self.request("stop")

    def trigger_killswitch(self, *, actor_user_id: str) -> dict[str, Any]:
        return self.request("trigger_killswitch", actor_user_id=actor_user_id)

    def queue_message(self, **message: Any) -> dict[str, Any]:
        return self.request("queue_message", **message)

    def publish_event(self, **event: Any) -> dict[str, Any]:
        return self.request("publish_event", **event)

    def events(self) -> list[dict[str, Any]]:
        result = self.request("events")
        events = result.get("events")
        return [dict(event) for event in events if isinstance(event, dict)] if isinstance(events, list) else []

    def desktop_poll(
        self,
        *,
        client_id: str,
        user: str,
        project_id: str = "",
        local_date: str = "",
        after_sequence: int = 0,
        after_ui_sequence: int = 0,
        daemon_id: str = "",
        client_capabilities: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self.request(
            "desktop_poll",
            client_id=client_id,
            user=user,
            project_id=project_id,
            local_date=local_date,
            after_sequence=after_sequence,
            after_ui_sequence=after_ui_sequence,
            daemon_id=daemon_id,
            client_capabilities=dict(client_capabilities or {}),
            limit=limit,
        )

    def acknowledge_desktop_delivery(self, *, client_id: str, outbox_message_id: str,
                                     attempt_count: int | None = None) -> dict[str, Any]:
        params = {"client_id": client_id, "outbox_message_id": outbox_message_id}
        if attempt_count is not None:
            params["attempt_count"] = attempt_count
        return self.request("desktop_ack_delivery", **params)

    def desktop_conversation_history(self, *, user: str, project_id: str = "", local_date: str = "", limit: int = 100) -> dict[str, Any]:
        return self.request("desktop_conversation_history", user=user, project_id=project_id, local_date=local_date, limit=limit)

    def mark_desktop_project_seen(self, *, user: str, project_id: str, through_sequence: int | None = None) -> dict[str, Any]:
        return self.request(
            "desktop_mark_project_seen",
            user=user,
            project_id=project_id,
            through_sequence=through_sequence,
        )

    def mark_desktop_conversation_seen(self, *, user: str, local_date: str = "", through_sequence: int | None = None) -> dict[str, Any]:
        return self.request("desktop_mark_conversation_seen", user=user, local_date=local_date, through_sequence=through_sequence)

    def projects(self, *, user: str) -> dict[str, Any]:
        return self.request("projects", user=user)

    def create_project(self, *, user: str, name: str, description: str, root_path: str, visibility: str = "private") -> dict[str, Any]:
        return self.request(
            "create_project", user=user, name=name, description=description, root_path=root_path, visibility=visibility,
        )

    def manageable_projects(self, *, user: str, status: str = "") -> dict[str, Any]:
        return self.request("manageable_projects", user=user, status=status)

    def project_context(self, *, user: str, project_id: str) -> dict[str, Any]:
        return self.request("project_context", user=user, project_id=project_id)

    def save_project_context(self, *, user: str, project_id: str, content: str) -> dict[str, Any]:
        return self.request("save_project_context", user=user, project_id=project_id, content=content)

    def project_config(self, *, user: str, project_id: str) -> dict[str, Any]:
        return self.request("project_config", user=user, project_id=project_id)

    def save_project_config(self, *, user: str, project_id: str, config: dict[str, Any]) -> dict[str, Any]:
        return self.request("save_project_config", user=user, project_id=project_id, config=config)

    def select_project_session(self, **values: Any) -> dict[str, Any]:
        return self.request("select_project_session", **values)

    def active_project_session(self, **values: Any) -> dict[str, Any]:
        return self.request("active_project_session", **values)

    def memory_sessions(self, **values: Any) -> dict[str, Any]:
        return self.request("memory_sessions", **values)

    def create_memory_session(self, **values: Any) -> dict[str, Any]:
        return self.request("create_memory_session", **values)

    def select_memory_session(self, **values: Any) -> dict[str, Any]:
        return self.request("select_memory_session", **values)

    def close_memory_session(self, **values: Any) -> dict[str, Any]:
        return self.request("close_memory_session", **values)

    def pending_questions(self, *, user: str) -> dict[str, Any]:
        return self.request("pending_questions", user=user)

    def answer_question(self, *, user: str, question_id: str, text: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("answer_question", user=user, question_id=question_id, text=text, payload=dict(payload or {}))

    def cancel_question(self, question_id: str, *, user: str) -> dict[str, Any]:
        return self.request("cancel_question", question_id=question_id, user=user)

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
        return self.request(
            "a2ui_action",
            client_id=client_id,
            user=user,
            surface_id=surface_id,
            source_component_id=source_component_id,
            action_name=action_name,
            context=dict(context or {}),
            data_model=dict(data_model or {}),
        )

    def integrations(self) -> dict[str, Any]:
        return self.request("integrations")

    def save_telegram_integration(self, *, user: str, values: dict[str, Any]) -> dict[str, Any]:
        return self.request("save_telegram_integration", user=user, values=dict(values))

    def save_discord_integration(self, *, user: str, values: dict[str, Any]) -> dict[str, Any]:
        return self.request("save_discord_integration", user=user, values=dict(values))

    def restart_integrations(self) -> dict[str, Any]:
        return self.request("restart_integrations")

    def inference_settings(self) -> dict[str, Any]:
        return self.request("inference_settings")

    def inference_providers(self) -> dict[str, Any]:
        return self.request("inference_providers")

    def inference_models(self, provider_key: str) -> dict[str, Any]:
        return self.request("inference_models", provider_key=provider_key)

    def set_inference_settings(self, *, provider_key: str, model_id: str) -> dict[str, Any]:
        # Model validation performs one bounded provider request. It needs a
        # longer timeout than routine daemon health and queue operations.
        client = V2DaemonClient(self.socket_path, timeout_sec=max(self.timeout_sec, 35.0))
        return client.request("set_inference_settings", provider_key=provider_key, model_id=model_id)

    def system_one_settings(self, *, actor_user_id: str) -> dict[str, Any]:
        return self.request("system_one_settings", actor_user_id=actor_user_id)

    def save_system_one_settings(self, *, actor_user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        client = V2DaemonClient(self.socket_path, timeout_sec=max(self.timeout_sec, 105.0))
        return client.request("save_system_one_settings", actor_user_id=actor_user_id, values=values)

    def web_tools_settings(self, *, actor_user_id: str) -> dict[str, Any]:
        return self.request("web_tools_settings", actor_user_id=actor_user_id)

    def code_mode_settings(self, *, actor_user_id: str) -> dict[str, Any]:
        return self.request("code_mode_settings", actor_user_id=actor_user_id)

    def save_code_mode_settings(self, *, actor_user_id: str, values: dict[str, Any], acknowledge_unsafe: bool = False) -> dict[str, Any]:
        return self.request("save_code_mode_settings", actor_user_id=actor_user_id, values=values, acknowledge_unsafe=acknowledge_unsafe)

    def verify_code_mode(self, *, actor_user_id: str) -> dict[str, Any]:
        return self.request("verify_code_mode", actor_user_id=actor_user_id)

    def save_web_tools_settings(self, *, actor_user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        return self.request("save_web_tools_settings", actor_user_id=actor_user_id, values=values)

    def verify_web_tools(self, *, actor_user_id: str, kind: str) -> dict[str, Any]:
        return self.request("verify_web_tools", actor_user_id=actor_user_id, kind=kind)

    def media_tools_settings(self, *, actor_user_id: str) -> dict[str, Any]:
        return self.request("media_tools_settings", actor_user_id=actor_user_id)

    def save_media_tools_settings(self, *, actor_user_id: str, kind: str, values: dict[str, Any]) -> dict[str, Any]:
        return self.request("save_media_tools_settings", actor_user_id=actor_user_id, kind=kind, values=values)

    def verify_media_tools(self, *, actor_user_id: str, kind: str, sample: str = "") -> dict[str, Any]:
        return self.request("verify_media_tools", actor_user_id=actor_user_id, kind=kind, sample=sample)

    def verify_stt_recording(self, *, actor_user_id: str, audio_base64: str, mime_type: str, duration_ms: int) -> dict[str, Any]:
        return self.request("verify_stt_recording", actor_user_id=actor_user_id, audio_base64=audio_base64, mime_type=mime_type, duration_ms=duration_ms)

    def agent_config_documents(self) -> dict[str, Any]:
        return self.request("agent_config_documents")

    def read_agent_config(self, file_name: str) -> dict[str, Any]:
        return self.request("read_agent_config", file_name=file_name)

    def save_agent_config(self, *, file_name: str, content: str) -> dict[str, Any]:
        return self.request("save_agent_config", file_name=file_name, content=content)

    def skills(self, *, actor_user_id: str) -> dict[str, Any]:
        return self.request("skills", actor_user_id=actor_user_id)

    def install_skill(self, *, actor_user_id: str, source_directory: str) -> dict[str, Any]:
        return self.request("install_skill", actor_user_id=actor_user_id, source_directory=source_directory)

    def customize_skill(self, *, actor_user_id: str, skill_id: str) -> dict[str, Any]:
        return self.request("customize_skill", actor_user_id=actor_user_id, skill_id=skill_id)

    def replace_skill(self, *, actor_user_id: str, skill_id: str, source_directory: str, resolution: str = "") -> dict[str, Any]:
        return self.request("replace_skill", actor_user_id=actor_user_id, skill_id=skill_id, source_directory=source_directory, resolution=resolution)

    def delete_skill(self, *, actor_user_id: str, skill_id: str) -> dict[str, Any]:
        return self.request("delete_skill", actor_user_id=actor_user_id, skill_id=skill_id)

    def scheduled_tasks(self, **filters: Any) -> dict[str, Any]:
        return self.request("scheduled_tasks", **filters)

    def scheduled_executions(self, **filters: Any) -> dict[str, Any]:
        return self.request("scheduled_executions", **filters)

    def deliveries(self, **filters: Any) -> dict[str, Any]:
        return self.request("deliveries", **filters)

    def retry_occurrence(self, occurrence_key: str) -> dict[str, Any]:
        return self.request("retry_occurrence", occurrence_key=occurrence_key)

    def pause_schedule(self, scheduled_task_id: str) -> dict[str, Any]:
        return self.request("pause_schedule", scheduled_task_id=scheduled_task_id)

    def cancel_schedule(self, scheduled_task_id: str) -> dict[str, Any]:
        return self.request("cancel_schedule", scheduled_task_id=scheduled_task_id)

    def resume_schedule(self, scheduled_task_id: str, *, actor_user_id: str = "") -> dict[str, Any]:
        return self.request("resume_schedule", scheduled_task_id=scheduled_task_id, actor_user_id=actor_user_id)


class V2DaemonServer:
    def __init__(self, daemon: Any, socket_path: str | Path | None = None, *, experimental_owner_ingress: bool = False,
                 experimental_authority_ingress: bool = False) -> None:
        if type(experimental_owner_ingress) is not bool or type(experimental_authority_ingress) is not bool:
            raise ValueError('owner_ingress_configuration_invalid')
        self.experimental_owner_ingress = experimental_owner_ingress
        # Test-only wiring: this is NOT a protected Desktop authentication
        # boundary and must not enable host Bash or be set by IPC clients.
        self.experimental_authority_ingress = experimental_authority_ingress
        self.daemon = daemon
        self.socket_path = Path(socket_path) if socket_path is not None else default_socket_path()
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._quiescence = threading.Condition()
        self._accepting_handlers = False
        self._listener_running = False
        self._handlers: set[threading.Thread] = set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.socket_path))
        bound_stat = self.socket_path.lstat()
        self._socket_identity = (bound_stat.st_dev, bound_stat.st_ino)
        os.chmod(self.socket_path, 0o600)
        self._server.listen(16)
        self._server.settimeout(0.5)
        self._stop.clear()
        with self._quiescence:
            self._accepting_handlers = True
            self._listener_running = True
            self._thread = threading.Thread(target=self._serve, name="alphonse-v2-ipc", daemon=True)
            try:
                self._thread.start()
            except Exception:
                self._accepting_handlers = False
                self._listener_running = False
                self._quiescence.notify_all()
                raise

    def stop(self) -> None:
        with self._quiescence:
            self._accepting_handlers = False
            self._quiescence.notify_all()
        if hasattr(self, '_oauth_loopback'):
            self._oauth_loopback.stop()
        self._stop.set()
        if self._server is not None:
            self._server.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        # A server may be stopped without having bound this path, or another
        # owner may have replaced the socket since it was bound. Never remove
        # an endpoint that this instance did not create.
        identity = self._socket_identity
        self._socket_identity = None
        if identity is not None:
            try:
                current = self.socket_path.lstat()
                if (current.st_dev, current.st_ino) == identity:
                    self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def is_serving(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def wait_quiescent(self, timeout: float | None = None) -> bool:
        """Wait until the listener and all accepted IPC requests have exited."""
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        with self._quiescence:
            done = self._quiescence.wait_for(
                lambda: not self._accepting_handlers and not self._listener_running and not self._handlers,
                timeout=timeout)
            oauth = getattr(self, '_oauth_loopback', None)
        if not done:
            return False
        return oauth is None or (callable(getattr(oauth, 'wait_quiescent', None))
                                 and oauth.wait_quiescent(None if deadline is None else max(0, deadline - time.monotonic())))

    def _serve(self) -> None:
        assert self._server is not None
        try:
            while not self._stop.is_set():
                try:
                    connection, _ = self._server.accept()
                except (TimeoutError, socket.timeout):
                    continue
                except OSError:
                    break
                with self._quiescence:
                    if not self._accepting_handlers:
                        connection.close()
                        continue
                    thread = threading.Thread(target=self._handle_tracked, args=(connection,), daemon=True)
                    self._handlers.add(thread)
                    try:
                        thread.start()
                    except Exception:
                        self._handlers.remove(thread)
                        connection.close()
                        self._quiescence.notify_all()
                        raise
        finally:
            with self._quiescence:
                self._listener_running = False
                self._quiescence.notify_all()

    def _handle_tracked(self, connection: socket.socket) -> None:
        try:
            self._handle(connection)
        finally:
            with self._quiescence:
                self._handlers.discard(threading.current_thread())
                self._quiescence.notify_all()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            request: Any = {}
            try:
                request = json.loads(_read_line(connection))
                if isinstance(request, dict) and request.get('method') == 'optional_management':
                    result = self._optional_management(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_ingress':
                    result = self._owner_ingress(connection, request)
                elif isinstance(request, dict) and request.get('method') in {
                        'experimental_owner_followup_add', 'experimental_owner_followup_list',
                        'experimental_owner_followup_remove', 'experimental_owner_composer_save',
                        'experimental_owner_composer_list', 'experimental_owner_composer_delete',
                        'experimental_owner_composer_reconcile', 'experimental_owner_steer_add',
                        'experimental_owner_steer_list', 'experimental_owner_steer_remove'}:
                    result = self._owner_followups(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_message_suggestion':
                    result = self._owner_message_suggestion(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_take_draft':
                    result = self._owner_take_draft(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_ack_draft':
                    result = self._owner_ack_draft(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_status':
                    result = self._owner_status(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_work_log':
                    result = self._owner_work_log(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_cancel_schedule':
                    result = self._owner_cancel_schedule(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'legacy_state_inventory':
                    result = self._legacy_state_inventory(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'legacy_state_items':
                    result = self._legacy_state_items(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_abandon':
                    result = self._owner_abandon(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_recovery_list':
                    result = self._owner_recovery_list(connection, request)
                elif isinstance(request, dict) and request.get('method') in {'experimental_owner_continuation_list', 'experimental_owner_continuation_choose'}:
                    result = self._owner_continuation(connection, request)
                elif isinstance(request, dict) and request.get('method') in {'get_owner_continuation_settings', 'set_owner_continuation_settings'}:
                    result = self._owner_continuation_settings(connection, request)
                elif isinstance(request, dict) and request.get('method') == 'experimental_owner_questions':
                    result = self._owner_questions(connection, request)
                elif isinstance(request, dict) and request.get('method') in {
                        'owner_reminder_create', 'owner_reminder_take', 'owner_reminder_ack',
                        'owner_reminder_list', 'owner_reminder_cancel', 'owner_reminder_resolve', 'owner_reminder_lookup'}:
                    result = self._owner_reminders(connection, request)
                elif isinstance(request, dict) and request.get('method') in {
                        'owner_scheduled_results_list', 'owner_scheduled_result_ack'}:
                    result = self._owner_scheduled_delivery(connection, request)
                elif isinstance(request, dict) and request.get('method') in {'legacy_notification_list', 'legacy_notification_archive'}:
                    result = self._legacy_notification_review(connection, request)
                else:
                    result = self._dispatch(request)
                response = {"version": PROTOCOL_VERSION, "request_id": _request_id(request), "ok": True, "result": result}
            except Exception as exc:
                response = {
                    "version": PROTOCOL_VERSION,
                    "request_id": _request_id(request),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            try:
                encoded = (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                # A malformed service result must not close the connection
                # without a protocol response or leak the malformed value.
                encoded = (json.dumps({"version": PROTOCOL_VERSION, "request_id": _request_id(request),
                    "ok": False, "error": "daemon_response_unavailable"}) + "\n").encode("utf-8")
            connection.sendall(encoded)

    def _owner_message_suggestion(self, connection, request):
        """Suggest only for the authenticated local owner's live request."""
        if not self.experimental_owner_ingress or getattr(self.daemon, 'experimental_owner_only', False) is not True:
            raise PermissionError('owner_suggestion_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        if not isinstance(params, dict) or set(params) != {'active_message_id', 'prompt'}:
            raise ValueError('owner_suggestion_payload_denied')
        message_id, prompt = params['active_message_id'], params['prompt']
        if (not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None
                or not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 4096):
            raise ValueError('owner_suggestion_payload_denied')
        uid = self._local_peer_uid(connection)
        users, queue = self.daemon.runtime.user_store, self.daemon.runtime.queue
        owner = users.admin_user()
        if (owner is None or owner.is_active is not True or owner.role not in ('admin', 'owner')
                or not users.is_admin(owner.user_id) or not users.db_path or users.db_path == ':memory:'
                or os.path.realpath(str(users.db_path)) != os.path.realpath(str(queue.db_path))):
            raise PermissionError('owner_suggestion_unavailable')
        if self.daemon.owner_message_status(message_id) != 'pending':
            return {'suggestion': None}
        with queue._connect() as conn:
            row = conn.execute("""SELECT q.prompt FROM v2_inbound_messages q
                JOIN v2_trusted_owner_ingress o ON o.message_id=q.message_id
                WHERE q.message_id=? AND q.user_id=? AND o.owner_id=? AND o.peer_uid=?""",
                (message_id, owner.user_id, owner.user_id, uid)).fetchone()
        if row is None:
            return {'suggestion': None}
        from alphonse.agent_v2.owner_message_suggestion import suggest
        suggestion = suggest(self.daemon.runtime.system_one_settings_store.get(), active_request=row['prompt'], draft=prompt)
        if self.daemon.owner_message_status(message_id) != 'pending':
            suggestion = None
        return {'suggestion': suggestion}

    def _owner_followups(self, connection, request):
        """Manage held follow-ups; this endpoint never dispatches work."""
        if not self.experimental_owner_ingress or getattr(self.daemon, 'experimental_owner_only', False) is not True:
            raise PermissionError('owner_followup_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        method, params = request.get('method'), request.get('params')
        expected = {'experimental_owner_followup_add': {'active_message_id', 'prompt', 'request_id'},
                    'experimental_owner_followup_list': set(),
                    'experimental_owner_followup_remove': {'id'},
                    'experimental_owner_composer_save': {'id', 'active_message_id', 'prompt'},
                    'experimental_owner_composer_list': set(),
                    'experimental_owner_composer_delete': {'id'},
                    'experimental_owner_composer_reconcile': {'id'},
                    'experimental_owner_steer_add': {'active_message_id','request_id','prompt'},
                    'experimental_owner_steer_list': set(),
                    'experimental_owner_steer_remove': {'id'}}[method]
        if not isinstance(params, dict) or set(params) != expected:
            raise ValueError('owner_followup_payload_denied')
        uid = self._local_peer_uid(connection)
        users = self.daemon.runtime.user_store
        owner = users.admin_user()
        if (owner is None or owner.is_active is not True or owner.role not in ('admin', 'owner')
                or not users.is_admin(owner.user_id)):
            raise PermissionError('owner_followup_unavailable')
        queue = self.daemon.runtime.queue
        if (not users.db_path or users.db_path == ':memory:'
                or os.path.realpath(str(users.db_path)) != os.path.realpath(str(queue.db_path))):
            raise PermissionError('owner_followup_unavailable')
        from alphonse.agent_v2.owner_followups import OwnerFollowups
        if method.startswith('experimental_owner_steer_'):
            from alphonse.agent_v2.owner_steering import submit, list_steering, remove_steering
            if method == 'experimental_owner_steer_add':
                return submit(queue, owner=owner.user_id, peer_uid=uid, daemon_id=self.daemon.daemon_id,
                              message_id=params['active_message_id'], request_id=params['request_id'], content=params['prompt'])
            if method == 'experimental_owner_steer_remove':
                return remove_steering(queue, owner=owner.user_id, peer_uid=uid, steering_id=params['id'])
            result = list_steering(queue, owner=owner.user_id, peer_uid=uid)
            for item in result['items']:
                status = self.daemon.owner_message_status(item['active_message_id'])
                item['request_status'] = status
                if item['status'] == 'accepted' and status in {'failed', 'unavailable', 'abandoned', 'started_unconfirmed'}:
                    item['status'] = 'needs_attention'
            return result
        if method.startswith('experimental_owner_composer_'):
            from alphonse.agent_v2.owner_composer_drafts import OwnerComposerDrafts
            drafts = OwnerComposerDrafts(queue, owner.user_id, uid)
            if method == 'experimental_owner_composer_save':
                return drafts.save(params['id'], params['active_message_id'], params['prompt'])
            if method == 'experimental_owner_composer_reconcile':
                return drafts.reconcile(params['id'])
            if method == 'experimental_owner_composer_delete':
                return drafts.delete_draft(params['id'])
            return drafts.list_drafts()
        followups = OwnerFollowups(queue, owner.user_id, uid)
        if method == 'experimental_owner_followup_add':
            return followups.add(params['active_message_id'], params['prompt'], params['request_id'])
        if method == 'experimental_owner_followup_remove':
            return followups.remove(params['id'])
        result = followups.list()
        for item in result['items']:
            if item['status'] == 'queued' and item['predecessor_active']:
                status = self.daemon.owner_message_status(item['predecessor_message_id'])
                if status in {'failed', 'unavailable', 'abandoned', 'started_unconfirmed'}:
                    item['status'] = 'needs_attention'
            if item['status'] == 'released':
                status = self.daemon.owner_message_status(item['id'])
                item['request_status'] = status
                if status in {'failed', 'unavailable', 'abandoned', 'started_unconfirmed'}:
                    item['status'] = 'needs_attention'
        return result

    def _owner_reminders(self, connection, request):
        """Restrict reminder mutation and pickup to the trusted local owner."""
        if not self.experimental_owner_ingress or getattr(self.daemon, 'experimental_owner_only', False) is not True:
            raise PermissionError('owner_reminder_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        method, params = request.get('method'), request.get('params')
        expected = {'owner_reminder_create': {'message', 'due_at', 'request_id'},
                    'owner_reminder_take': set(), 'owner_reminder_ack': {'id', 'token'},
                    'owner_reminder_list': set(),
                    'owner_reminder_cancel': {'id'}, 'owner_reminder_resolve': {'id'},
                    'owner_reminder_lookup': {'request_id'}}[method]
        if not isinstance(params, dict) or set(params) != expected:
            raise ValueError('owner_reminder_payload_denied')
        self._local_peer_uid(connection)
        users = self.daemon.runtime.user_store
        owner = users.admin_user()
        if owner is None or owner.is_active is not True or owner.role not in ('admin', 'owner') or not users.is_admin(owner.user_id):
            raise PermissionError('owner_reminder_unavailable')
        from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
        queue = self.daemon.runtime.queue
        if (type(queue) is not SQLiteMessageQueue or queue.experimental_claim_fencing is not True
                or not users.db_path or users.db_path == ':memory:'
                or os.path.realpath(str(users.db_path)) != os.path.realpath(str(queue.db_path))):
            raise PermissionError('owner_reminder_unavailable')
        from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
        reminders = OwnerDesktopReminders(str(users.db_path))
        if method == 'owner_reminder_create':
            return reminders.create(owner.user_id, params['message'], params['due_at'], params['request_id'])
        if method == 'owner_reminder_take':
            return reminders.take_due(owner.user_id)
        if method == 'owner_reminder_list':
            return {'reminders': reminders.list_open(owner.user_id)}
        if method == 'owner_reminder_cancel':
            return reminders.cancel(owner.user_id, params['id'])
        if method == 'owner_reminder_resolve':
            return reminders.resolve_unconfirmed(owner.user_id, params['id'])
        if method == 'owner_reminder_lookup':
            return reminders.lookup(owner.user_id, params['request_id'])
        return reminders.acknowledge(owner.user_id, params['id'], params['token'])

    def _owner_scheduled_delivery(self, connection, request):
        """Only the authenticated local owner can present/ack their results."""
        if not self.experimental_owner_ingress or getattr(self.daemon, 'experimental_owner_only', False) is not True:
            raise PermissionError('owner_scheduled_delivery_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        method, params = request.get('method'), request.get('params')
        expected = set() if method == 'owner_scheduled_results_list' else {'id'}
        if not isinstance(params, dict) or set(params) != expected:
            raise ValueError('owner_scheduled_delivery_payload_denied')
        self._local_peer_uid(connection)
        users = self.daemon.runtime.user_store
        owner = users.admin_user()
        if owner is None or owner.is_active is not True or owner.role not in ('admin', 'owner') or not users.is_admin(owner.user_id):
            raise PermissionError('owner_scheduled_delivery_unavailable')
        from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
        queue = self.daemon.runtime.queue
        if (type(queue) is not SQLiteMessageQueue or queue.experimental_claim_fencing is not True
                or not users.db_path or users.db_path == ':memory:'
                or os.path.realpath(str(users.db_path)) != os.path.realpath(str(queue.db_path))):
            raise PermissionError('owner_scheduled_delivery_unavailable')
        from alphonse.agent_v2.owner_scheduled_delivery import OwnerScheduledDelivery
        deliveries = OwnerScheduledDelivery(str(users.db_path))
        if method == 'owner_scheduled_results_list':
            return {'results': deliveries.list_pending(owner.user_id)}
        return deliveries.acknowledge(owner.user_id, params['id'])

    def _owner_ingress(self, connection, request):
        if not self.experimental_owner_ingress:
            raise PermissionError('owner_ingress_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        allowed_fields = {'prompt', 'reply_to_message_id'}
        if self.experimental_authority_ingress:
            allowed_fields |= {'message_id', 'authority_request_id'}
        if (not isinstance(params, dict) or 'prompt' not in params
                or set(params) - allowed_fields):
            raise ValueError('owner_ingress_payload_denied')
        binding = None
        if self.experimental_authority_ingress:
            if any(not isinstance(params.get(key), str) or re.fullmatch(r'[0-9a-f]{32}', params[key]) is None
                   for key in ('message_id', 'authority_request_id')):
                raise ValueError('owner_authority_binding_invalid')
            binding = (params['message_id'], params['authority_request_id'])
        if ('reply_to_message_id' in params and
                (not isinstance(params['reply_to_message_id'], str)
                 or re.fullmatch(r'[0-9a-f]{32}', params['reply_to_message_id']) is None)):
            raise ValueError('owner_question_context_unavailable')
        uid = self._local_peer_uid(connection)
        if (getattr(self.daemon, 'experimental_owner_only', False) is True
                and not self.daemon.owner_processing_ready()):
            raise RuntimeError('owner_worker_unavailable')
        from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
        queue = self.daemon.runtime.queue
        if type(queue) is not SQLiteMessageQueue:
            raise PermissionError('owner_ingress_unavailable')
        return queue.enqueue_trusted_local_owner(prompt=params['prompt'], peer_uid=uid,
                    reply_to_message_id=params.get('reply_to_message_id'), return_receipt=True,
                    _authority_binding=binding)

    def _legacy_notification_review(self, connection, request):
        """Review legacy deliveries as the authenticated local owner only."""
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        method = request.get('method')
        expected = set() if method == 'legacy_notification_list' else {'message_id', 'review_token'}
        if not isinstance(params, dict) or set(params) != expected:
            raise ValueError('legacy_notification_payload_denied')
        self._local_peer_uid(connection)
        users = self.daemon.runtime.user_store
        owner = users.admin_user()
        if (owner is None or owner.is_active is not True or owner.role not in ('admin', 'owner')
                or not users.is_admin(owner.user_id)):
            raise PermissionError('legacy_notification_owner_unavailable')
        outbox = self.daemon.runtime.outbox
        if method == 'legacy_notification_list':
            return {'items': outbox.list_legacy_tui_owner_items(owner.user_id)}
        if not isinstance(params['message_id'], str) or not isinstance(params['review_token'], str):
            raise ValueError('legacy_notification_payload_denied')
        return outbox.archive_legacy_tui_owner_item(owner.user_id, params['message_id'], params['review_token'])

    def _legacy_state_inventory(self, connection, request):
        """Return bounded owner-only counts; never replay or change stored work."""
        runtime, queue, actor = self._legacy_state_database(connection, request)
        from alphonse.agent_v2.core.messages.legacy_state_inventory import inspect_legacy_state
        try:
            same_schedule_db = os.path.samefile(queue.db_path, runtime.schedule_store.db_path)
        except (AttributeError, OSError, TypeError, ValueError):
            same_schedule_db = False
        result = inspect_legacy_state(queue.db_path, expected_owner=actor,
                                      include_schedules=same_schedule_db)
        return {'status': 'available', 'inventory': result} if result is not None else {'status': 'unavailable'}

    def _legacy_state_items(self, connection, request):
        """Return bounded owner-only metadata; never change or resume work."""
        runtime, queue, actor = self._legacy_state_database(connection, request)
        from alphonse.agent_v2.core.messages.legacy_state_items import inspect_legacy_items
        try:
            same_schedule_db = os.path.samefile(queue.db_path, runtime.schedule_store.db_path)
        except (AttributeError, OSError, TypeError, ValueError):
            same_schedule_db = False
        result = inspect_legacy_items(queue.db_path, expected_owner=actor, include_schedules=same_schedule_db)
        return {'status': 'available', 'items': result} if result is not None else {'status': 'unavailable'}

    def _legacy_state_database(self, connection, request):
        if not self.experimental_owner_ingress or getattr(self.daemon, 'experimental_owner_only', False) is not True:
            raise PermissionError('legacy_state_inventory_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        if (not isinstance(params, dict) or set(params) != {'actor_user_id'}
                or not isinstance(params['actor_user_id'], str) or not params['actor_user_id'].strip()):
            raise ValueError('legacy_state_inventory_payload_denied')
        if self._local_peer_uid(connection) != os.getuid():
            raise PermissionError('legacy_state_inventory_unavailable')
        from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
        runtime = self.daemon.runtime
        queue = getattr(runtime, 'queue', None)
        try:
            if (type(queue) is not SQLiteMessageQueue or queue.experimental_claim_fencing is not True
                    or not queue.db_path or queue.db_path == ':memory:'
                    or any(not os.path.samefile(queue.db_path, store.db_path) for store in
                           (runtime.user_store, runtime.question_store, runtime.outbox))):
                raise ValueError
        except (AttributeError, OSError, TypeError, ValueError):
            raise PermissionError('legacy_state_inventory_unavailable') from None
        return runtime, queue, params['actor_user_id']

    def _owner_cancel_schedule(self, connection, request):
        if (not self.experimental_owner_ingress or getattr(self.daemon, 'experimental_owner_only', False) is not True
                or request.get('version') != PROTOCOL_VERSION):
            raise PermissionError('owner_schedule_cancel_unavailable')
        params = request.get('params')
        if (not isinstance(params, dict) or set(params) != {'scheduled_task_id'}
                or not isinstance(params['scheduled_task_id'], str) or not params['scheduled_task_id'].strip()):
            raise ValueError('owner_schedule_cancel_payload_denied')
        self._local_peer_uid(connection)
        actor = self.daemon.runtime.user_store.admin_user()
        if actor is None or not self.daemon.runtime.user_store.is_admin(actor.user_id):
            raise PermissionError('owner_schedule_cancel_unavailable')
        return self.daemon.cancel_owner_schedule(actor_user_id=actor.user_id,
                                                  scheduled_task_id=params['scheduled_task_id'])

    def _owner_status(self, connection, request):
        if not self.experimental_owner_ingress:
            raise PermissionError('owner_status_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        if not isinstance(params, dict) or set(params) not in (set(), {'message_id'}):
            raise ValueError('owner_status_payload_denied')
        self._local_peer_uid(connection)
        if not params:
            reason = self.daemon.owner_processing_blocking_reason()
            return {'owner_only': getattr(self.daemon, 'experimental_owner_only', False) is True,
                    'processing_ready': reason is None,
                    'continuation_available': reason is None and getattr(self.daemon, '_continuation_experimental_enabled', False) is True,
                    'blocking_reason': reason,
                    'paused_legacy_requests': self.daemon.owner_paused_legacy_requests()}
        return self.daemon.owner_message_status_payload(params['message_id'])

    def _owner_recovery_list(self, connection, request):
        if not self.experimental_owner_ingress:
            raise PermissionError('owner_recovery_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        if request.get('params') != {}:
            raise ValueError('owner_recovery_payload_denied')
        self._local_peer_uid(connection)
        return {'items': self.daemon.owner_recovery_list()}

    def _owner_work_log(self, connection, request):
        if not self.experimental_owner_ingress:
            raise PermissionError('owner_work_log_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        if not isinstance(params, dict) or set(params) != {'message_id'}:
            raise ValueError('owner_work_log_unavailable')
        self._local_peer_uid(connection)
        from .core.intelligence.simple_loop.owner_work_log import owner_work_log
        try:
            return owner_work_log(self.daemon.runtime.queue, params['message_id'])
        except Exception:
            # Do not expose database paths or checkpoint content over IPC.
            raise ValueError('owner_work_log_unavailable') from None

    def _owner_continuation(self, connection, request):
        if (not self.experimental_owner_ingress or not getattr(self.daemon, 'experimental_owner_only', False)
                or not getattr(self.daemon, '_continuation_experimental_enabled', False)):
            raise PermissionError('owner_continuation_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        self._local_peer_uid(connection)
        if request.get('method') == 'experimental_owner_continuation_list':
            if params != {}:
                raise ValueError('owner_continuation_payload_denied')
            return {'items': self.daemon.owner_continuation_list()}
        if (not isinstance(params, dict) or set(params) != {'message_id', 'checkpoint_version', 'approve'}
                or not isinstance(params['message_id'], str)
                or type(params['checkpoint_version']) is not int or params['checkpoint_version'] < 1
                or type(params['approve']) is not bool):
            raise ValueError('owner_continuation_payload_denied')
        return self.daemon.owner_continuation_choose(params['message_id'], params['checkpoint_version'], params['approve'])

    def _owner_continuation_settings(self, connection, request):
        if not self.experimental_owner_ingress or not getattr(self.daemon, 'experimental_owner_only', False):
            raise PermissionError('owner_continuation_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        self._local_peer_uid(connection)
        params = request.get('params')
        if request.get('method') == 'get_owner_continuation_settings':
            if params != {}:
                raise ValueError('owner_continuation_payload_denied')
            return self.daemon.owner_continuation_settings()
        return self.daemon.set_owner_continuation_settings(params)

    def _owner_abandon(self, connection, request):
        if not self.experimental_owner_ingress:
            raise PermissionError('owner_abandon_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        if not isinstance(params, dict) or set(params) != {'message_id'}:
            raise ValueError('owner_abandon_payload_denied')
        self._local_peer_uid(connection)
        return self.daemon.abandon_owner_message(params['message_id'])

    def _owner_take_draft(self, connection, request):
        # The OS peer and server-held checkpoint identify authority; a client
        # cannot supply an actor, claim receipt, destination, or draft bytes.
        if not self.experimental_owner_ingress:
            raise PermissionError('owner_desktop_pickup_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        if not isinstance(params, dict) or set(params) != {'message_id'}:
            raise ValueError('owner_desktop_pickup_payload_denied')
        self._local_peer_uid(connection)
        return self.daemon.take_owner_desktop_draft(params['message_id'])

    def _owner_questions(self, connection, request):
        if not self.experimental_owner_ingress:
            raise PermissionError('owner_question_context_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        if request.get('params') != {}:
            raise ValueError('owner_question_context_unavailable')
        self._local_peer_uid(connection)
        return {'questions': self.daemon.owner_delivered_questions()}

    def _owner_ack_draft(self, connection, request):
        if not self.experimental_owner_ingress:
            raise PermissionError('owner_desktop_ack_unavailable')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        params = request.get('params')
        if not isinstance(params, dict) or set(params) != {'message_id', 'acknowledgement_token'}:
            raise ValueError('owner_desktop_ack_payload_denied')
        self._local_peer_uid(connection)
        self.daemon.acknowledge_owner_desktop_draft(params['message_id'], params['acknowledgement_token'])
        return {'status': 'client_acknowledged'}

    @staticmethod
    def _local_peer_uid(connection):
        # The owner-only Unix socket and kernel peer identity authenticate the
        # local OS principal. Payload user/actor fields are never credentials.
        import struct
        if hasattr(connection, 'getpeereid'):
            uid, _ = connection.getpeereid()
        elif sys.platform == 'darwin':
            # CPython on macOS exposes LOCAL_PEERCRED but not getpeereid.
            # Use the OS getpeereid function rather than trusting a payload UID.
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            getpeereid = libc.getpeereid
            getpeereid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
            getpeereid.restype = ctypes.c_int
            peer_uid, peer_gid = ctypes.c_uint(), ctypes.c_uint()
            if getpeereid(connection.fileno(), ctypes.byref(peer_uid), ctypes.byref(peer_gid)) != 0:
                raise PermissionError('management_peer_auth_failed')
            uid = peer_uid.value
        elif hasattr(socket, 'SO_PEERCRED'):
            _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        else:
            raise PermissionError('management_peer_auth_unavailable')
        if uid != os.getuid():
            raise PermissionError('management_peer_denied')
        return uid

    def _optional_management(self, connection, request):
        from alphonse.agent_v2.accounts import AccountSession
        self._local_peer_uid(connection)
        params = request.get('params', {})
        if not isinstance(params, dict) or set(params) - {'action', 'values'}:
            raise ValueError('management_identity_payload_denied')
        if request.get('version') != PROTOCOL_VERSION:
            raise ValueError('daemon_protocol_version_unsupported')
        owner_only = getattr(self.daemon, 'experimental_owner_only', False) is True
        if owner_only:
            users = self.daemon.runtime.user_store
            try:
                if not os.path.samefile(users.db_path, self.daemon.runtime.integration_store.db_path):
                    raise PermissionError('management_owner_database_unavailable')
            except (OSError, AttributeError, TypeError, ValueError):
                raise PermissionError('management_owner_database_unavailable') from None
            owner = users.admin_user()
            if (owner is None or owner.is_active is not True or owner.role not in ('admin', 'owner')
                    or not users.is_admin(owner.user_id)):
                raise PermissionError('management_owner_unavailable')
            session = AccountSession(owner.user_id)
        else:
            session = AccountSession(self.daemon.runtime.user)
        action = params.get('action')
        values = params.get('values', {})
        if not isinstance(values, dict):
            raise ValueError('management_values_invalid')
        if owner_only:
            if action not in {'catalog', 'begin', 'save_grant', 'revoke_grant', 'pause', 'disconnect', 'retry_deletion'}:
                raise PermissionError('owner_only_account_action_unavailable')
            if action == 'catalog' and values:
                raise ValueError('management_values_invalid')
            if action == 'begin' and (values.get('provider') != 'google' or values.get('capabilities') != ['mail_read']):
                raise PermissionError('owner_only_account_scope_unavailable')
            if action == 'save_grant' and (values.get('beneficiary_user_id') != owner.user_id
                    or values.get('operations') != ['mail_read'] or values.get('resources') != ['INBOX']
                    or values.get('recipient_user_ids') != [owner.user_id] or values.get('project_id') is not None):
                raise PermissionError('owner_only_account_scope_unavailable')
            if action == 'revoke_grant':
                grant_id = values.get('grant_id')
                with self.daemon.runtime.integration_store._connect() as db:
                    row = db.execute('SELECT connection_id FROM v2_account_grants WHERE grant_id=?', (grant_id,)).fetchone()
                grant_record = self.daemon.runtime.integration_store.get(row['connection_id']) if row is not None else None
                if (grant_record is None or grant_record.provider_key != 'google'
                        or grant_record.connection.get('owner_user_id') != owner.user_id):
                    raise PermissionError('owner_only_account_scope_unavailable')
            if action in {'begin', 'save_grant', 'pause', 'disconnect', 'retry_deletion'}:
                connection_id = values.get('connection_id')
                record = self.daemon.runtime.integration_store.get(connection_id) if isinstance(connection_id, str) else None
                if record is not None and (record.provider_key != 'google'
                        or record.connection.get('owner_user_id') != owner.user_id):
                    raise PermissionError('owner_only_account_scope_unavailable')
                if (action == 'begin' and record is not None
                        and set(record.connection.get('capabilities', ())) - {'mail_read'}):
                    # A narrow reconnect would silently replace broader consent.
                    raise PermissionError('owner_only_combined_account_reconnect_unavailable')
        oauth = self.daemon.optional_accounts(session)
        management = self.daemon.runtime.integration_management
        management.oauth = oauth
        if action == 'begin':
            from alphonse.agent_v2.integrations.oauth_loopback import OAuthLoopback
            # A previously accepted request must not create a new callback
            # listener after shutdown has begun.
            with self._quiescence:
                if not self._accepting_handlers:
                    raise RuntimeError('daemon_stopping')
                if not hasattr(self, '_oauth_loopback'):
                    self._oauth_loopback = OAuthLoopback(oauth)
                self._oauth_loopback.start(values['provider'])
        # Callback exchange is accepted only by the loopback listener, never raw IPC.
        if action == 'complete':
            raise PermissionError('oauth_callback_transport_required')
        result = management.request(session, action, values)
        if owner_only and action == 'catalog':
            result['owner_gmail_setup_only'] = True
            result['providers'] = [dict(provider='google', capabilities=['mail_read'])
                                   for provider in result['providers'] if provider['provider'] == 'google'
                                   and 'mail_read' in provider['capabilities']]
            result['connections'] = [item for item in result['connections'] if item['provider'] == 'google']
            connection_ids = {item['connection_id'] for item in result['connections']}
            result['grants'] = [grant for grant in result['grants'] if grant['connection_id'] in connection_ids]
        return result

    def _dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("daemon_request_object_required")
        version = request.get("version")
        if version is not None and int(version) != PROTOCOL_VERSION:
            raise ValueError(f"daemon_protocol_version_unsupported: {version}")
        method = str(request.get("method") or "").strip()
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        if (getattr(self.daemon, 'experimental_owner_only', False) is True
                and method in {'desktop_ack_delivery', 'answer_question', 'cancel_question', 'a2ui_action',
                               'publish_event', 'register_event_worker', 'create_event_automation',
                               'retry_occurrence', 'resume_schedule', 'approve_access_request',
                               'save_telegram_integration', 'save_discord_integration'}):
            raise RuntimeError('owner_only_operation_unavailable')
        if method == "ping":
            return {"service": "alphonse-v2-daemon", "status": "ready"}
        if method == "runtime_identity":
            if params:
                raise ValueError('runtime_identity_payload_denied')
            return self.daemon.runtime_identity()
        if method == "get_trusted_local_bash_settings":
            if params:
                raise ValueError('trusted_local_bash_setting_invalid')
            return self.daemon.trusted_local_bash_settings()
        if method == "set_trusted_local_bash_settings":
            if set(params) != {'enabled'} or type(params['enabled']) is not bool:
                raise ValueError('trusted_local_bash_setting_invalid')
            return self.daemon.set_trusted_local_bash_settings(params['enabled'])
        if method == "status":
            runtime = self.daemon.runtime
            status_error = ""
            try:
                due_count = len(runtime.schedule_store.due_tasks())
            except Exception as exc:
                due_count = -1
                status_error = f"schedule_status: {type(exc).__name__}: {exc}"
            try:
                queue_size = runtime.queue.size()
            except Exception as exc:
                queue_size = -1
                status_error = status_error or f"queue_status: {type(exc).__name__}: {exc}"
            try:
                outbound_queue_size = len(runtime.outbox.list_pending(limit=1000))
            except Exception as exc:
                outbound_queue_size = -1
                status_error = status_error or f"outbox_status: {type(exc).__name__}: {exc}"
            queue_counts = getattr(runtime.queue, "status_counts", lambda: {})()
            outbox_counts = getattr(runtime.outbox, "status_counts", lambda: {})()
            db_path = str(getattr(runtime.user_store, "db_path", ":memory:"))
            persistence = storage_metrics(db_path) if db_path != ":memory:" else {"database_path": ":memory:"}
            processor_thread = getattr(self.daemon, "_processor_thread", None)
            return {
                "service": "alphonse-v2-daemon",
                "daemon_id": getattr(self.daemon, "daemon_id", ""),
                "queue_size": queue_size,
                "outbound_queue_size": outbound_queue_size,
                "inbound_counts": dict(queue_counts) if isinstance(queue_counts, dict) else {},
                "outbound_counts": dict(outbox_counts) if isinstance(outbox_counts, dict) else {},
                "active_work": self.daemon.active_work(),
                "activity": self.daemon.activity_status(),
                "due_schedules": due_count,
                "processor_alive": bool(processor_thread is not None and processor_thread.is_alive()),
                "last_processor_error": str(getattr(self.daemon, "_last_processor_error", "") or ""),
                "status_error": status_error,
                "scheduler": self.daemon.scheduler.stats.__dict__,
                "persistence": persistence,
                "intelligence_engine": runtime.intelligence_engine_settings_store.get().to_dict(),
            }
        if method == "stop":
            threading.Thread(target=self.daemon.stop, name="alphonse-v2-stop", daemon=True).start()
            return {"status": "stopping"}
        if method == "trigger_killswitch":
            return self.daemon.trigger_killswitch(
                actor_user_id=str(params.get("actor_user_id") or ""),
                source={"integration_id": "desktop", "provider_key": "desktop"},
            )
        if method == "decision_context":
            return self.daemon.decision_context(
                actor_user_id=str(params.get("actor_user_id") or ""),
                task_id=str(params.get("task_id") or ""),
                decision_id=str(params.get("decision_id") or ""),
                source_id=str(params.get("source_id") or ""),
            )
        if method == "events":
            return {"events": self.daemon.pop_activity_events()}
        if method == "desktop_poll":
            return self.daemon.poll_desktop(
                client_id=str(params.get("client_id") or "desktop"),
                user=str(params.get("user") or "local"),
                project_id=str(params.get("project_id") or ""),
                local_date=str(params.get("local_date") or ""),
                after_sequence=int(params.get("after_sequence") or 0),
                after_ui_sequence=int(params.get("after_ui_sequence") or 0),
                daemon_id=str(params.get("daemon_id") or ""),
                client_capabilities=params.get("client_capabilities") if isinstance(params.get("client_capabilities"), dict) else {},
                limit=int(params.get("limit") or 100),
            )
        if method == "desktop_ack_delivery":
            return {
                "acknowledged": self.daemon.acknowledge_desktop_delivery(
                    client_id=str(params.get("client_id") or "desktop"),
                    outbox_message_id=str(params.get("outbox_message_id") or ""),
                    **({"attempt_count": params["attempt_count"]} if "attempt_count" in params else {}),
                )
            }
        if method == "desktop_conversation_history":
            return {
                "messages": self.daemon.desktop_conversation_history(
                    user=str(params.get("user") or "local"),
                    project_id=str(params.get("project_id") or ""),
                    local_date=str(params.get("local_date") or ""),
                    limit=int(params.get("limit") or 100),
                )
            }
        if method == "desktop_mark_project_seen":
            through = params.get("through_sequence")
            return self.daemon.mark_desktop_project_seen(
                user=str(params.get("user") or "local"),
                project_id=str(params.get("project_id") or ""),
                through_sequence=int(through) if through is not None else None,
            )
        if method == "desktop_mark_conversation_seen":
            through = params.get("through_sequence")
            return self.daemon.mark_desktop_project_seen(
                user=str(params.get("user") or "local"),
                local_date=str(params.get("local_date") or ""),
                through_sequence=int(through) if through is not None else None,
            )
        if method == "restart_integrations":
            self.daemon.restart_integrations()
            return {"status": "restarted"}
        if method == "inference_settings":
            return {"settings": self.daemon.runtime.inference_settings_store.get().to_dict()}
        if method == "inference_providers":
            from alphonse.agent_v2.inference_settings import inference_provider_descriptors
            from alphonse.agent_v2.inference_settings import provider_status

            return {
                "providers": [
                    provider_status(descriptor.provider_key)
                    for descriptor in inference_provider_descriptors()
                ]
            }
        if method == "inference_models":
            from alphonse.agent_v2.inference_settings import provider_status

            provider_key = str(params.get("provider_key") or "").strip()
            return provider_status(provider_key)
        if method == "set_inference_settings":
            settings = self.daemon.update_inference_settings(
                provider_key=str(params.get("provider_key") or ""),
                model_id=str(params.get("model_id") or ""),
            )
            return {"settings": settings}
        if method == "agent_config_documents":
            return {"documents": self.daemon.list_agent_config()}
        if method == "read_agent_config":
            return {"document": self.daemon.read_agent_config(str(params.get("file_name") or ""))}
        if method == "save_agent_config":
            return {
                "document": self.daemon.save_agent_config(
                    str(params.get("file_name") or ""),
                    str(params.get("content") or ""),
                )
            }
        if method == "onboarding_status":
            return self.daemon.onboarding_status()
        if method == "onboard":
            return self.daemon.onboard(display_name=str(params.get("display_name") or ""), workspace_root=str(params.get("workspace_root") or ""), users_root=str(params.get("users_root") or ""), import_v1=bool(params.get("import_v1")))
        if method == "current_user":
            return self.daemon.current_user()
        if method == "settings":
            return self.daemon.settings()
        if method == "save_settings":
            mirror = params.get("mirror_automation_messages_to_preferred_channel")
            return self.daemon.save_settings(
                actor_user_id=str(params.get("actor_user_id") or ""),
                workspace_root=str(params.get("workspace_root") or ""),
                timezone_name=str(params.get("timezone") or ""),
                mirror_automation_messages_to_preferred_channel=bool(mirror) if mirror is not None else None,
            )
        if method == "timezone_settings":
            return self.daemon.timezone_settings(actor_user_id=str(params.get("actor_user_id") or ""))
        if method == "save_timezone_settings":
            return self.daemon.save_timezone_settings(actor_user_id=str(params.get("actor_user_id") or ""), timezone_name=str(params.get("timezone") or ""))
        if method == "memory_settings":
            return {"settings": self.daemon.memory_settings(actor_user_id=str(params.get("actor_user_id") or ""))}
        if method == "save_memory_settings":
            return {"settings": self.daemon.save_memory_settings(actor_user_id=str(params.get("actor_user_id") or ""), values=dict(params.get("values") or {}))}
        if method == "system_one_settings":
            return {"settings": self.daemon.system_one_settings(actor_user_id=str(params.get("actor_user_id") or ""))}
        if method == "save_system_one_settings":
            return {"settings": self.daemon.save_system_one_settings(actor_user_id=str(params.get("actor_user_id") or ""), values=dict(params.get("values") or {}))}
        if method == "intelligence_engine_settings":
            return {"settings": self.daemon.intelligence_engine_settings(actor_user_id=str(params.get("actor_user_id") or ""))}
        if method == "save_intelligence_engine_settings":
            return {"settings": self.daemon.save_intelligence_engine_settings(actor_user_id=str(params.get("actor_user_id") or ""), values=dict(params.get("values") or {}))}
        if method == "legacy_skill_migration":
            location = params.get("location")
            if location is not None and not isinstance(location, dict):
                raise ValueError("legacy_location_invalid")
            return self.daemon.legacy_skill_migration(actor_user_id=str(params.get("actor_user_id") or ""), operation=str(params.get("operation") or "preview"), artifact_id=str(params.get("artifact_id") or ""), review_token=str(params.get("review_token") or ""), instructions=str(params.get("instructions") or ""), location=location)
        if method in {"artifacts", "relocate_artifact", "reconcile_artifacts", "update_artifact", "set_artifact_enabled", "delete_artifact", "unassigned_artifacts", "attach_artifact_to_skill"}:
            raise ValueError("artifact_catalog_retired: use legacy_skill_migration preview/apply/rollback; no files were changed")
        if method == "skills_overview":
            return self.daemon.list_skills_overview(actor_user_id=str(params.get("actor_user_id") or ""))
        if method == "skills":
            return self.daemon.list_skills_workspace(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""))
        if method == "skill_appearance":
            return self.daemon.skill_appearance(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""))
        if method == "set_skill_appearance":
            return self.daemon.set_skill_appearance(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""), preset=str(params.get("preset") or ""))
        if method == "create_skill":
            return {"skill": self.daemon.create_skill(actor_user_id=str(params.get("actor_user_id") or ""), name=str(params.get("name") or ""), description=str(params.get("description") or ""), instructions=str(params.get("instructions") or ""))}
        if method == "update_skill":
            return {"skill": self.daemon.update_skill(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""), description=str(params.get("description") or ""))}
        if method == "read_skill_file":
            return self.daemon.read_skill_file(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""), path=str(params.get("path") or ""))
        if method == "customize_skill":
            return {"skill": self.daemon.customize_skill(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""))}
        if method == "write_skill_file":
            return self.daemon.write_skill_file(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""), path=str(params.get("path") or ""), content=str(params.get("content") or ""))
        if method == "delete_skill_file":
            return self.daemon.delete_skill_file(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""), path=str(params.get("path") or ""))
        if method == "add_skill_file":
            return self.daemon.add_skill_file(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""), source_path=str(params.get("source_path") or ""))
        if method == "install_skill":
            return {"skill": self.daemon.install_skill(actor_user_id=str(params.get("actor_user_id") or ""), source_directory=str(params.get("source_directory") or ""))}
        if method == "export_skill":
            return self.daemon.export_skill(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""), destination_directory=str(params.get("destination_directory") or ""))
        if method == "replace_skill":
            return {"skill": self.daemon.replace_skill(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""), source_directory=str(params.get("source_directory") or ""), resolution=str(params.get("resolution") or ""))}
        if method == "delete_skill":
            return self.daemon.delete_skill(actor_user_id=str(params.get("actor_user_id") or ""), skill_id=str(params.get("skill_id") or ""))
        if method == "web_tools_settings":
            return {"settings": self.daemon.web_tools_settings(actor_user_id=str(params.get("actor_user_id") or ""))}
        if method == "code_mode_settings":
            return {"settings": self.daemon.code_mode_settings(actor_user_id=str(params.get("actor_user_id") or ""))}
        if method == "save_code_mode_settings":
            return {"settings": self.daemon.save_code_mode_settings(actor_user_id=str(params.get("actor_user_id") or ""), values=dict(params.get("values") or {}), acknowledge_unsafe=bool(params.get("acknowledge_unsafe")))}
        if method == "verify_code_mode":
            return self.daemon.verify_code_mode(actor_user_id=str(params.get("actor_user_id") or ""))
        if method == "save_web_tools_settings":
            return {"settings": self.daemon.save_web_tools_settings(actor_user_id=str(params.get("actor_user_id") or ""), values=dict(params.get("values") or {}))}
        if method == "verify_web_tools":
            return {"result": self.daemon.verify_web_tools(actor_user_id=str(params.get("actor_user_id") or ""), kind=str(params.get("kind") or ""))}
        if method == "media_tools_settings":
            return {"settings": self.daemon.media_tools_settings(actor_user_id=str(params.get("actor_user_id") or ""))}
        if method == "save_media_tools_settings":
            return {"settings": self.daemon.save_media_tools_settings(actor_user_id=str(params.get("actor_user_id") or ""), kind=str(params.get("kind") or ""), values=dict(params.get("values") or {}))}
        if method == "verify_media_tools":
            return self.daemon.verify_media_tools(actor_user_id=str(params.get("actor_user_id") or ""), kind=str(params.get("kind") or ""), sample=str(params.get("sample") or ""))
        if method == "verify_stt_recording":
            return self.daemon.verify_stt_recording(
                actor_user_id=str(params.get("actor_user_id") or ""),
                audio_base64=str(params.get("audio_base64") or ""),
                mime_type=str(params.get("mime_type") or ""),
                duration_ms=int(params.get("duration_ms") or 0),
            )
        if method == "users":
            return {"users": self.daemon.list_users()}
        if method == "create_user":
            return {"user": self.daemon.create_user(display_name=str(params.get("display_name") or ""), role=str(params.get("role") or "member"))}
        if method == "update_user":
            return {"user": self.daemon.update_user(str(params.get("user_id") or ""), display_name=params.get("display_name"), role=params.get("role"), is_active=params.get("is_active"))}
        if method == "delete_user":
            return self.daemon.delete_user(str(params.get("user_id") or ""), confirmation=str(params.get("confirmation") or ""))
        if method == "user_context":
            return self.daemon.user_context(str(params.get("user_id") or ""))
        if method == "save_user_context":
            return self.daemon.save_user_context(str(params.get("user_id") or ""), str(params.get("content") or ""))
        if method == "bind_user_address":
            return {"address": self.daemon.bind_user_address(**dict(params))}
        if method == "remove_user_address":
            return {"removed": self.daemon.remove_user_address(str(params.get("address_id") or ""))}
        if method == "set_user_aliases":
            aliases = params.get("aliases") if isinstance(params.get("aliases"), list) else []
            return self.daemon.set_user_aliases(user_id=str(params.get("user_id") or ""), aliases=[str(item) for item in aliases])
        if method == "pending_access_requests":
            return {"requests": self.daemon.pending_access_requests()}
        if method == "approve_access_request":
            return self.daemon.approve_access_request(
                request_id=str(params.get("request_id") or ""),
                display_name=str(params.get("display_name") or ""),
                user_id=str(params.get("user_id") or ""),
            )
        if method == "reject_access_request":
            return {"request": self.daemon.reject_access_request(request_id=str(params.get("request_id") or ""))}
        if method == "projects":
            return {"projects": self.daemon.list_projects(user=str(params.get("user") or ""))}
        if method == "project_recent_files":
            return {
                "files": self.daemon.project_recent_files(
                    user=str(params.get("user") or ""),
                    project_id=str(params.get("project_id") or ""),
                    limit=int(params.get("limit") or 4),
                )
            }
        if method == "project_files":
            return {"files": self.daemon.project_files(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""))}
        if method == "remove_project_file":
            return self.daemon.remove_project_file(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""), name=str(params.get("name") or ""))
        if method == "copy_desktop_project_files":
            source_paths = params.get("source_paths") if isinstance(params.get("source_paths"), list) else []
            return {
                "files": self.daemon.copy_desktop_project_files(
                    user=str(params.get("user") or ""),
                    project_id=str(params.get("project_id") or ""),
                    source_paths=[str(item) for item in source_paths],
                )
            }
        if method == "household_overview":
            return self.daemon.household_overview(
                actor_user_id=str(params.get("actor_user_id") or ""),
                include_personal=params.get("include_personal") is True,
            )
        if method == "work_items":
            return self.daemon.work_items(
                actor_user_id=str(params.get("actor_user_id") or ""),
                owner_user_id=str(params.get("owner_user_id") or ""),
                view=str(params.get("view") or "active"), project_id=str(params.get("project_id") or ""),
                limit=int(params.get("limit") or 50), offset=int(params.get("offset") or 0),
            )
        if method == "project_records":
            current = self.daemon.current_user().get("user")
            actor = str(current.get("user_id") or "") if isinstance(current, dict) else ""
            return self.daemon.project_records(
                actor_user_id=actor, owner_user_id=str(params.get("owner_user_id") or ""),
                project_id=str(params.get("project_id") or ""),
                schedules_offset=int(params.get("schedules_offset") or 0), executions_offset=int(params.get("executions_offset") or 0), todos_offset=int(params.get("todos_offset") or 0),
                limit=int(params.get("limit") or 50),
            )
        if method == "mutate_project_todo":
            current = self.daemon.current_user().get("user")
            actor = str(current.get("user_id") or "") if isinstance(current, dict) else ""
            return {"todo": self.daemon.mutate_project_todo(actor_user_id=actor, project_id=str(params.get("project_id") or ""),
                action=str(params.get("action") or ""), title=str(params.get("title") or ""),
                todo_id=str(params.get("todo_id") or ""), request_key=str(params.get("request_key") or ""))}
        if method == "manageable_projects":
            return {"projects": self.daemon.manageable_projects(user=str(params.get("user") or ""), status=str(params.get("status") or ""))}
        if method == "create_project":
            return {
                "project": self.daemon.create_project(
                    user=str(params.get("user") or "local"),
                    name=str(params.get("name") or ""),
                    description=str(params.get("description") or ""),
                    root_path=str(params.get("root_path") or ""),
                    visibility=str(params.get("visibility") or "private"),
                )
            }
        if method == "import_project":
            return {"project": self.daemon.import_project(user=str(params.get("user") or "local"), name=str(params.get("name") or ""), description=str(params.get("description") or ""), root_path=str(params.get("root_path") or ""), visibility=str(params.get("visibility") or "private"))}
        if method == "update_project":
            members = params.get("member_user_ids")
            return {"project": self.daemon.update_project(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""), name=str(params.get("name") or ""), description=str(params.get("description") or ""), visibility=str(params.get("visibility") or "private"), member_user_ids=[str(item) for item in members] if isinstance(members, list) else None)}
        if method == "set_project_status":
            return {"project": self.daemon.set_project_status(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""), status=str(params.get("status") or ""))}
        if method == "project_config":
            return {"config": self.daemon.project_config(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""))}
        if method == "save_project_config":
            config = params.get("config")
            return {"project": self.daemon.save_project_config(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""), config=config if isinstance(config, dict) else {})}
        if method == "archive_project":
            return {"project": self.daemon.archive_project(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""))}
        if method == "restore_project":
            return {"project": self.daemon.restore_project(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""))}
        if method == "delete_project":
            return self.daemon.delete_project(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""), confirmation=str(params.get("confirmation") or ""))
        if method == "project_members":
            return {"members": self.daemon.project_members(str(params.get("project_id") or ""))}
        if method == "add_project_member":
            self.daemon.add_project_member(str(params.get("project_id") or ""), str(params.get("user_id") or ""))
            return {"added": True}
        if method == "remove_project_member":
            return {"removed": self.daemon.remove_project_member(str(params.get("project_id") or ""), str(params.get("user_id") or ""))}
        if method == "project_context":
            return self.daemon.read_project_context(
                user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""),
            )
        if method == "save_project_context":
            return {
                "project": self.daemon.save_project_context(
                    user=str(params.get("user") or "local"),
                    project_id=str(params.get("project_id") or ""),
                    content=str(params.get("content") or ""),
                )
            }
        if method == "select_project_session":
            return {
                "session": self.daemon.select_project_session(
                    user=str(params.get("user") or "local"),
                    integration_id=str(params.get("integration_id") or "desktop"),
                    channel_target=str(params.get("channel_target") or params.get("user") or "local"),
                    thread_id=str(params.get("thread_id") or ""),
                    project_id=str(params.get("project_id") or ""),
                )
            }
        if method == "active_project_session":
            return {
                "session": self.daemon.active_project_session(
                    user=str(params.get("user") or "local"),
                    integration_id=str(params.get("integration_id") or "desktop"),
                    channel_target=str(params.get("channel_target") or params.get("user") or "local"),
                    thread_id=str(params.get("thread_id") or ""),
                )
            }
        if method == "memory_sessions":
            common = {
                "user": str(params.get("user") or "local"), "project_id": str(params.get("project_id") or ""),
                "integration_id": str(params.get("integration_id") or "desktop"),
                "channel_target": str(params.get("channel_target") or params.get("user") or "local"),
                "thread_id": str(params.get("thread_id") or ""),
            }
            return {
                "sessions": self.daemon.list_memory_sessions(user=common["user"], project_id=common["project_id"], include_closed=bool(params.get("include_closed"))),
                "active_session": self.daemon.active_memory_session(**common),
            }
        if method in {"create_memory_session", "select_memory_session", "close_memory_session"}:
            common = {
                "user": str(params.get("user") or "local"), "project_id": str(params.get("project_id") or ""),
                "integration_id": str(params.get("integration_id") or "desktop"),
                "channel_target": str(params.get("channel_target") or params.get("user") or "local"),
                "thread_id": str(params.get("thread_id") or ""),
            }
            if method == "create_memory_session":
                return {"session": self.daemon.create_memory_session(name=str(params.get("name") or ""), **common)}
            if method == "select_memory_session":
                return {"session": self.daemon.select_memory_session(session_id=str(params.get("session_id") or ""), **common)}
            return {"session": self.daemon.close_memory_session(session_id=str(params.get("session_id") or ""), **common)}
        if method == "memory_migration_status":
            return {"migration": self.daemon.memory_migration_status(user=str(params.get("user") or "local"), project_id=str(params.get("project_id") or ""))}
        if method == "pending_questions":
            user = self.daemon._question_actor(str(params.get("user") or ""))
            return {"questions": [question.to_dict() for question in self.daemon.runtime.question_store.list_pending_for_respondent(user)]}
        if method == "answer_question":
            payload = params.get("payload") if isinstance(params.get("payload"), dict) else {}
            return self.daemon.answer_question(
                user=str(params.get("user") or "local"),
                question_id=str(params.get("question_id") or ""),
                text=str(params.get("text") or ""),
                payload=dict(payload),
            )
        if method == "cancel_question":
            return {"cancelled": self.daemon.cancel_question(str(params.get("question_id") or ""), user=str(params.get("user") or ""))}
        if method == "a2ui_action":
            return self.daemon.a2ui_action(
                client_id=str(params.get("client_id") or "desktop"),
                user=str(params.get("user") or "local"),
                surface_id=str(params.get("surface_id") or ""),
                source_component_id=str(params.get("source_component_id") or ""),
                action_name=str(params.get("action_name") or ""),
                context=params.get("context") if isinstance(params.get("context"), dict) else {},
                data_model=params.get("data_model") if isinstance(params.get("data_model"), dict) else {},
            )
        if method == "integrations":
            return {"integrations": self.daemon.list_integrations()}
        if method == "save_telegram_integration":
            values = params.get("values") if isinstance(params.get("values"), dict) else {}
            return {
                "integration": self.daemon.save_telegram_integration(
                    user=str(params.get("user") or "local"), values=dict(values),
                )
            }
        if method == "save_discord_integration":
            values = params.get("values") if isinstance(params.get("values"), dict) else {}
            return {
                "integration": self.daemon.save_discord_integration(
                    user=str(params.get("user") or "local"), values=dict(values),
                )
            }
        if method == "queue_message":
            if getattr(self.daemon, 'experimental_owner_only', False) is True:
                raise RuntimeError('owner_only_use_owner_conversation')
            integration_id = str(params.get("integration_id") or "desktop")
            provider_key = str(params.get("provider_key") or "desktop")
            user = str(params.get("user") or "")
            if provider_key == "tui" or integration_id in {"tui", "desktop"}:
                current = self.daemon.current_user().get("user")
                if isinstance(current, dict):
                    user = str(current.get("user_id") or "")
            return self.daemon.ingest_message(
                prompt=str(params.get("prompt") or ""),
                user=user,
                project_id=str(params.get("project_id") or ""),
                tag=str(params.get("tag") or ""),
                correlation_id=str(params.get("correlation_id") or ""),
                metadata=dict(params.get("metadata") or {}),
                integration_id=integration_id,
                provider_key=provider_key,
                provider_user_id=str(params.get("provider_user_id") or ""),
                channel_target=str(params.get("channel_target") or ""),
                provider_message_id=str(params.get("provider_message_id") or ""),
            )
        if method == "publish_event":
            return self.daemon.publish_event(**params)
        if method == "register_event_worker":
            return {"worker": self.daemon.register_event_worker(**params)}
        if method == "register_event_type":
            return {"event_type": self.daemon.register_event_type(**params)}
        if method == "create_event_automation":
            return {"automation": self.daemon.create_event_automation(**params)}
        if method == "automation_catalog":
            return self.daemon.automation_catalog()
        if method == "expand_automation_event":
            return {"event": self.daemon.expand_automation_event(str(params.get("event_key") or ""))}
        if method == "agenda":
            from alphonse.agent_v2.agenda import agenda
            return agenda(self.daemon.runtime, actor=str(params.get("actor_user_id") or ""),
                          start_day=str(params.get("start_day") or ""), end_day=str(params.get("end_day") or ""))
        if method == "scheduled_tasks":
            status = str(params.get("status") or "").strip() or None
            return {"tasks": self.daemon.scheduled_tasks(
                actor_user_id=str(params.get("actor_user_id") or params.get("user") or ""),
                owner_user_id=str(params.get("owner_user_id") or ""),
                status=status,
                limit=int(params.get("limit") or 100),
            )}
        if method == "scheduled_executions":
            scheduled_task_id = str(params.get("scheduled_task_id") or "").strip()
            if not scheduled_task_id:
                raise ValueError("scheduled_task_id_required")
            executions = self.daemon.scheduled_task_executions(
                actor_user_id=str(params.get("actor_user_id") or params.get("user") or ""),
                scheduled_task_id=scheduled_task_id,
                limit=int(params.get("limit") or 100),
            )
            return {"executions": executions}
        if method == "deliveries":
            from alphonse.agent_v2.core.io import OutboundSelector

            status = str(params.get("status") or "").strip() or None
            deliveries = self.daemon.runtime.outbox.list(
                OutboundSelector(
                    integration_id=str(params.get("integration_id") or "").strip() or None,
                    channel_target=str(params.get("channel_target") or "").strip() or None,
                    status=status,
                    correlation_id=str(params.get("correlation_id") or "").strip() or None,
                    audience_user_id=str(params.get("audience_user_id") or "").strip() or None,
                ),
                limit=int(params.get("limit") or 100),
            )
            return {"deliveries": [delivery.to_dict() for delivery in deliveries]}
        if method == "retry_occurrence":
            occurrence_key = str(params.get("occurrence_key") or "").strip()
            if not occurrence_key:
                raise ValueError("occurrence_key_required")
            ok = self.daemon.runtime.schedule_store.mark_occurrence_retry(
                occurrence_key,
                worker_id="",
                error="manual_retry",
                next_attempt_at=datetime.now(timezone.utc).isoformat(),
            )
            return {"status": "queued" if ok else "not_found", "occurrence_key": occurrence_key}
        if method == "pause_schedule":
            return {"task": self.daemon.pause_scheduled_task(actor_user_id=str(params.get("actor_user_id") or params.get("user") or ""), scheduled_task_id=str(params.get("scheduled_task_id") or ""))}
        if method == "resume_schedule":
            return {"task": self.daemon.resume_scheduled_task(actor_user_id=str(params.get("actor_user_id") or params.get("user") or ""), scheduled_task_id=str(params.get("scheduled_task_id") or ""))}
        if method == "cancel_schedule":
            return {"task": self.daemon.cancel_scheduled_task(actor_user_id=str(params.get("actor_user_id") or params.get("user") or ""), scheduled_task_id=str(params.get("scheduled_task_id") or ""))}
        if method == "update_scheduled_task":
            return {"task": self.daemon.update_scheduled_task(actor_user_id=str(params.get("actor_user_id") or params.get("user") or ""), scheduled_task_id=str(params.get("scheduled_task_id") or ""), name=str(params.get("name") or ""), prompt=str(params.get("prompt") or ""))}
        if method == "delete_scheduled_task":
            return {"deleted": self.daemon.delete_scheduled_task(actor_user_id=str(params.get("actor_user_id") or params.get("user") or ""), scheduled_task_id=str(params.get("scheduled_task_id") or ""))}
        raise ValueError(f"daemon_method_not_found: {method}")


def _read_line(connection: socket.socket) -> str:
    chunks: list[bytes] = []
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    return b"".join(chunks).split(b"\n", 1)[0].decode("utf-8")


def _request_id(request: Any) -> str:
    return str(request.get("request_id") or "") if isinstance(request, dict) else ""
