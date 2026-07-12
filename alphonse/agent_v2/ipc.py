"""Local Unix-socket protocol for v2 daemon clients."""

from __future__ import annotations

import json
import os
import socket
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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

    def stop(self) -> dict[str, Any]:
        return self.request("stop")

    def queue_message(self, **message: Any) -> dict[str, Any]:
        return self.request("queue_message", **message)

    def events(self) -> list[dict[str, Any]]:
        result = self.request("events")
        events = result.get("events")
        return [dict(event) for event in events if isinstance(event, dict)] if isinstance(events, list) else []

    def desktop_poll(
        self,
        *,
        client_id: str,
        user: str,
        after_sequence: int = 0,
        after_ui_sequence: int = 0,
        client_capabilities: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self.request(
            "desktop_poll",
            client_id=client_id,
            user=user,
            after_sequence=after_sequence,
            after_ui_sequence=after_ui_sequence,
            client_capabilities=dict(client_capabilities or {}),
            limit=limit,
        )

    def acknowledge_desktop_delivery(self, *, client_id: str, outbox_message_id: str) -> dict[str, Any]:
        return self.request("desktop_ack_delivery", client_id=client_id, outbox_message_id=outbox_message_id)

    def projects(self, *, user: str) -> dict[str, Any]:
        return self.request("projects", user=user)

    def create_project(self, *, user: str, name: str, description: str, root_path: str, visibility: str = "private") -> dict[str, Any]:
        return self.request(
            "create_project", user=user, name=name, description=description, root_path=root_path, visibility=visibility,
        )

    def project_context(self, *, user: str, project_id: str) -> dict[str, Any]:
        return self.request("project_context", user=user, project_id=project_id)

    def save_project_context(self, *, user: str, project_id: str, content: str) -> dict[str, Any]:
        return self.request("save_project_context", user=user, project_id=project_id, content=content)

    def select_project_session(self, **values: Any) -> dict[str, Any]:
        return self.request("select_project_session", **values)

    def active_project_session(self, **values: Any) -> dict[str, Any]:
        return self.request("active_project_session", **values)

    def pending_questions(self, *, user: str) -> dict[str, Any]:
        return self.request("pending_questions", user=user)

    def answer_question(self, *, user: str, question_id: str, text: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("answer_question", user=user, question_id=question_id, text=text, payload=dict(payload or {}))

    def cancel_question(self, question_id: str) -> dict[str, Any]:
        return self.request("cancel_question", question_id=question_id)

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

    def agent_config_documents(self) -> dict[str, Any]:
        return self.request("agent_config_documents")

    def read_agent_config(self, file_name: str) -> dict[str, Any]:
        return self.request("read_agent_config", file_name=file_name)

    def save_agent_config(self, *, file_name: str, content: str) -> dict[str, Any]:
        return self.request("save_agent_config", file_name=file_name, content=content)

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


class V2DaemonServer:
    def __init__(self, daemon: Any, socket_path: str | Path | None = None) -> None:
        self.daemon = daemon
        self.socket_path = Path(socket_path) if socket_path is not None else default_socket_path()
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
        os.chmod(self.socket_path, 0o600)
        self._server.listen(16)
        self._server.settimeout(0.5)
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="alphonse-v2-ipc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._server.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            request: Any = {}
            try:
                request = json.loads(_read_line(connection))
                result = self._dispatch(request)
                response = {"version": PROTOCOL_VERSION, "request_id": _request_id(request), "ok": True, "result": result}
            except Exception as exc:
                traceback.print_exc()
                response = {
                    "version": PROTOCOL_VERSION,
                    "request_id": _request_id(request),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            connection.sendall((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))

    def _dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("daemon_request_object_required")
        version = request.get("version")
        if version is not None and int(version) != PROTOCOL_VERSION:
            raise ValueError(f"daemon_protocol_version_unsupported: {version}")
        method = str(request.get("method") or "").strip()
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        if method == "ping":
            return {"service": "alphonse-v2-daemon", "status": "ready"}
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
            }
        if method == "stop":
            threading.Thread(target=self.daemon.stop, name="alphonse-v2-stop", daemon=True).start()
            return {"status": "stopping"}
        if method == "events":
            return {"events": self.daemon.pop_activity_events()}
        if method == "desktop_poll":
            return self.daemon.poll_desktop(
                client_id=str(params.get("client_id") or "desktop"),
                user=str(params.get("user") or "local"),
                after_sequence=int(params.get("after_sequence") or 0),
                after_ui_sequence=int(params.get("after_ui_sequence") or 0),
                client_capabilities=params.get("client_capabilities") if isinstance(params.get("client_capabilities"), dict) else {},
                limit=int(params.get("limit") or 100),
            )
        if method == "desktop_ack_delivery":
            return {
                "acknowledged": self.daemon.acknowledge_desktop_delivery(
                    client_id=str(params.get("client_id") or "desktop"),
                    outbox_message_id=str(params.get("outbox_message_id") or ""),
                )
            }
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
        if method == "projects":
            return {"projects": self.daemon.list_projects(user=str(params.get("user") or "local"))}
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
                    integration_id=str(params.get("integration_id") or "tui"),
                    channel_target=str(params.get("channel_target") or params.get("user") or "local"),
                    thread_id=str(params.get("thread_id") or ""),
                    project_id=str(params.get("project_id") or ""),
                )
            }
        if method == "active_project_session":
            return {
                "session": self.daemon.active_project_session(
                    user=str(params.get("user") or "local"),
                    integration_id=str(params.get("integration_id") or "tui"),
                    channel_target=str(params.get("channel_target") or params.get("user") or "local"),
                    thread_id=str(params.get("thread_id") or ""),
                )
            }
        if method == "pending_questions":
            user = str(params.get("user") or "local")
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
            return {"cancelled": self.daemon.cancel_question(str(params.get("question_id") or ""))}
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
        if method == "queue_message":
            return self.daemon.ingest_message(
                prompt=str(params.get("prompt") or ""),
                user=str(params.get("user") or ""),
                project_id=str(params.get("project_id") or ""),
                tag=str(params.get("tag") or ""),
                correlation_id=str(params.get("correlation_id") or ""),
                metadata=dict(params.get("metadata") or {}),
                integration_id=str(params.get("integration_id") or "tui"),
                provider_key=str(params.get("provider_key") or "tui"),
                provider_user_id=str(params.get("provider_user_id") or ""),
                channel_target=str(params.get("channel_target") or ""),
                provider_message_id=str(params.get("provider_message_id") or ""),
            )
        if method == "scheduled_tasks":
            status = str(params.get("status") or "").strip() or None
            tasks = self.daemon.runtime.schedule_store.list_tasks(
                owner_user_id=str(params.get("owner_user_id") or "").strip() or None,
                project_id=str(params.get("project_id") or "").strip() or None,
                status=status,
                limit=int(params.get("limit") or 100),
            )
            return {"tasks": [task.to_dict() for task in tasks]}
        if method == "scheduled_executions":
            scheduled_task_id = str(params.get("scheduled_task_id") or "").strip()
            if not scheduled_task_id:
                raise ValueError("scheduled_task_id_required")
            executions = self.daemon.runtime.schedule_store.list_executions(
                scheduled_task_id=scheduled_task_id,
                limit=int(params.get("limit") or 100),
            )
            return {"executions": [execution.to_dict() for execution in executions]}
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
            task = self.daemon.runtime.schedule_store.pause_task(str(params.get("scheduled_task_id") or ""))
            return {"task": task.to_dict()}
        if method == "cancel_schedule":
            task = self.daemon.runtime.schedule_store.cancel_task(str(params.get("scheduled_task_id") or ""))
            return {"task": task.to_dict()}
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
