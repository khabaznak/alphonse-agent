"""Local Unix-socket protocol for v2 daemon clients."""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Any


def default_socket_path() -> Path:
    return Path(os.getenv("ALPHONSE_V2_SOCKET_PATH") or Path.home() / ".alphonse" / "v2-daemon.sock")


class V2DaemonClient:
    def __init__(self, socket_path: str | Path | None = None, *, timeout_sec: float = 2.0) -> None:
        self.socket_path = Path(socket_path) if socket_path is not None else default_socket_path()
        self.timeout_sec = max(0.1, float(timeout_sec))

    def request(self, method: str, **params: Any) -> dict[str, Any]:
        payload = json.dumps({"method": str(method), "params": params}, sort_keys=True) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_sec)
            connection.connect(str(self.socket_path))
            connection.sendall(payload.encode("utf-8"))
            data = _read_line(connection)
        response = json.loads(data)
        if not isinstance(response, dict):
            raise RuntimeError("daemon_invalid_response")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "daemon_request_failed"))
        return dict(response.get("result") or {})

    def ping(self) -> dict[str, Any]:
        return self.request("ping")

    def status(self) -> dict[str, Any]:
        return self.request("status")

    def queue_message(self, **message: Any) -> dict[str, Any]:
        return self.request("queue_message", **message)

    def events(self) -> list[dict[str, Any]]:
        result = self.request("events")
        events = result.get("events")
        return [dict(event) for event in events if isinstance(event, dict)] if isinstance(events, list) else []

    def restart_integrations(self) -> dict[str, Any]:
        return self.request("restart_integrations")


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
            try:
                request = json.loads(_read_line(connection))
                result = self._dispatch(request)
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            connection.sendall((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))

    def _dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("daemon_request_object_required")
        method = str(request.get("method") or "").strip()
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        if method == "ping":
            return {"service": "alphonse-v2-daemon", "status": "ready"}
        if method == "status":
            runtime = self.daemon.runtime
            return {
                "service": "alphonse-v2-daemon",
                "queue_size": runtime.queue.size(),
                "scheduler": self.daemon.scheduler.stats.__dict__,
            }
        if method == "events":
            return {"events": self.daemon.pop_activity_events()}
        if method == "restart_integrations":
            self.daemon.restart_integrations()
            return {"status": "restarted"}
        if method == "queue_message":
            queued = self.daemon.runtime.channel.queue_message(
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
            return {"message_id": queued.message_id}
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
