"""Run bounded Python programs that call V2 tools through a Unix-socket bridge."""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from alphonse.agent_v2.code_mode_settings import CodeModeSettings
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService

DOCKER_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TOOL_CALLS = 16
MAX_PARALLEL_CALLS = 4


class ProgramRunner:
    """Docker-only runner; no local subprocess fallback is intentionally provided."""

    def __init__(self, *, docker_bin: str = "docker", image: str = DOCKER_IMAGE, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, settings_provider: Callable[[], CodeModeSettings] | None = None) -> None:
        self._settings_provider = settings_provider
        self._fallback_settings = CodeModeSettings(enabled=True, docker_bin=docker_bin, image=image, timeout_seconds=timeout_seconds)
        self._availability: bool | None = None
        self._availability_checked_at = 0.0
        self._availability_key: tuple[str, bool, bool] | None = None

    def available(self) -> bool:
        settings = self._settings()
        return self._available_for(settings)

    def _available_for(self, settings: CodeModeSettings) -> bool:
        if not settings.available:
            return False
        key = (settings.docker_bin, settings.enabled, settings.verification_ready)
        if self._availability_key == key and self._availability is not None and time.monotonic() - self._availability_checked_at < 30:
            return self._availability
        try:
            probe = subprocess.run([settings.docker_bin, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=3, check=False)
        except (OSError, subprocess.TimeoutExpired):
            self._availability = False
            self._availability_checked_at = time.monotonic()
            self._availability_key = key
            return False
        self._availability = probe.returncode == 0 and bool(str(probe.stdout or "").strip())
        self._availability_checked_at = time.monotonic()
        self._availability_key = key
        return self._availability

    def run(self, *, source: str, invocation_service: ToolInvocationService) -> dict[str, Any]:
        settings = self._settings()
        if not self._available_for(settings):
            return _runner_failure("docker_unavailable", "Docker is required for program execution and is not available.")
        with tempfile.TemporaryDirectory(prefix="alphonse-program-") as root_text:
            root = Path(root_text)
            bridge_dir = root / "bridge"
            app_dir = root / "app"
            bridge_dir.mkdir(mode=0o777)
            app_dir.mkdir()
            (app_dir / "program.py").write_text(source, encoding="utf-8")
            (app_dir / "runner.py").write_text(_CONTAINER_RUNNER, encoding="utf-8")
            (app_dir / "alphonse_tools.py").write_text(_CONTAINER_TOOLS, encoding="utf-8")

            token = secrets.token_urlsafe(32)
            socket_path = bridge_dir / "tools.sock"
            events: list[dict[str, Any]] = []
            server = _BridgeServer(socket_path, token, invocation_service, events, max_tool_calls=settings.max_tool_calls, max_parallel_calls=settings.max_parallel_calls)
            server.start()
            command = [settings.docker_bin, "run", "--rm"]
            if settings.network_disabled: command += ["--network", "none"]
            if settings.read_only_filesystem: command += ["--read-only"]
            if settings.drop_all_capabilities: command += ["--cap-drop", "ALL"]
            if settings.no_new_privileges: command += ["--security-opt", "no-new-privileges"]
            command += ["--pids-limit", str(settings.pid_limit), "--memory", f"{settings.memory_mb}m", "--cpus", str(settings.cpu_count)]
            if settings.run_as_non_root: command += ["--user", "65534:65534"]
            command += ["--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={settings.tmpfs_mb}m",
                "-e", f"ALPHONSE_BRIDGE_TOKEN={token}",
                "-v", f"{bridge_dir}:/bridge:rw", "-v", f"{app_dir}:/app:ro", "-w", "/tmp",
                settings.image, "python", "/app/runner.py",
            ]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=settings.timeout_seconds, check=False)
            except subprocess.TimeoutExpired:
                return {**_runner_failure("program_timeout", f"Program exceeded {settings.timeout_seconds:g} seconds."), "tool_calls": events}
            finally:
                server.stop()

            payload = _final_payload(completed.stdout)
            if completed.returncode != 0:
                return {**_runner_failure("container_failed", str(completed.stderr or completed.stdout or "Container failed.")), "tool_calls": events}
            if payload is None:
                return {**_runner_failure("program_result_invalid", "Program must return one JSON object."), "tool_calls": events}
            if payload.get("kind") == "interrupted":
                return {"status": "waiting", "program_result": payload.get("result"), "tool_calls": events}
            result = payload.get("result")
            if not isinstance(result, dict):
                return {**_runner_failure("program_result_invalid", "Program main() must return a JSON object."), "tool_calls": events}
            return {"status": "success", "program_result": result, "tool_calls": events}

    def verify(self) -> dict[str, object]:
        """Run the configured image's Python entrypoint without starting a program phase."""
        settings = self._settings()
        try:
            probe = subprocess.run([settings.docker_bin, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=3, check=False)
            if probe.returncode != 0:
                return {"ready": False, "error": str(probe.stderr or "Docker daemon is unavailable.").strip()}
            image = subprocess.run([settings.docker_bin, "run", "--rm", "--network", "none", settings.image, "python", "--version"], capture_output=True, text=True, timeout=30, check=False)
            if image.returncode != 0:
                return {"ready": False, "error": str(image.stderr or image.stdout or "Configured image could not run Python.").strip()}
            return {"ready": True, "preview": str(image.stdout or image.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ready": False, "error": str(exc)}

    def _settings(self) -> CodeModeSettings:
        return self._settings_provider() if self._settings_provider is not None else self._fallback_settings


class _BridgeServer:
    def __init__(self, path: Path, token: str, service: ToolInvocationService, events: list[dict[str, Any]], *, max_tool_calls: int = MAX_TOOL_CALLS, max_parallel_calls: int = MAX_PARALLEL_CALLS) -> None:
        self.path, self.token, self.service, self.events = path, token, service, events
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        os.chmod(path, 0o777)
        self.max_tool_calls, self.max_parallel_calls = max_tool_calls, max_parallel_calls
        self._server.listen(max_parallel_calls)
        self._server.settimeout(0.2)
        self._stop = threading.Event()
        self._count = 0
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_parallel_calls)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._server.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._server.accept()
            except TimeoutError:
                continue
            self._pool.submit(self._handle, connection)

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            stream = connection.makefile("rwb")
            try:
                request = json.loads(stream.readline().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return
            response = self._invoke(request)
            stream.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            stream.flush()

    def _invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("id") or uuid4())
        if request.get("token") != self.token:
            return {"id": request_id, "result": _failed_result("bridge_auth_failed", "Bridge authentication failed.")}
        tool_id = str(request.get("tool_id") or "")
        arguments = request.get("arguments")
        if not tool_id or not isinstance(arguments, dict):
            return {"id": request_id, "result": _failed_result("bridge_request_invalid", "tool_id and object arguments are required.")}
        with self._lock:
            self._count += 1
            if self._count > self.max_tool_calls:
                return {"id": request_id, "result": _failed_result("program_call_budget_exhausted", "Program exceeded its tool-call budget.")}
        result = self.service.invoke(tool_id, arguments, call_id=f"program-call-{request_id}", parallel=bool(request.get("parallel")))
        self.events.append(result)
        return {"id": request_id, "result": result}


def _runner_failure(code: str, message: str) -> dict[str, Any]:
    return {"status": "failed", "program_result": None, "error": {"code": code, "message": message}, "tool_calls": []}


def _failed_result(code: str, message: str) -> dict[str, Any]:
    return {"call_id": "", "tool_id": "", "status": "failed", "result": None, "error": {"code": code, "message": message}}


def _final_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(str(stdout or "").splitlines()):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("kind") in {"final", "interrupted"}:
            return payload
    return None


_CONTAINER_RUNNER = '''import asyncio, importlib.util, json
from alphonse_tools import Tools, ProgramInterrupted
spec = importlib.util.spec_from_file_location("program", "/app/program.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
async def run():
    try:
        result = await module.main(Tools())
        print(json.dumps({"kind": "final", "result": result}, ensure_ascii=False))
    except ProgramInterrupted as exc:
        print(json.dumps({"kind": "interrupted", "result": exc.result}, ensure_ascii=False))
asyncio.run(run())
'''

_CONTAINER_TOOLS = '''import asyncio, json, os, socket, uuid
class ProgramInterrupted(Exception):
    def __init__(self, result): self.result = result
class Tools:
    async def call(self, tool_id, arguments):
        result = await asyncio.to_thread(self._request, tool_id, arguments, False)
        if result.get("status") == "waiting": raise ProgramInterrupted(result)
        return result
    async def gather(self, calls):
        return await asyncio.gather(*[self._call_parallel(item) for item in calls])
    async def _call_parallel(self, item):
        result = await asyncio.to_thread(self._request, item["tool_id"], item.get("arguments", {}), True)
        if result.get("status") == "waiting": raise ProgramInterrupted(result)
        return result
    def _request(self, tool_id, arguments, parallel):
        request = {"id": uuid.uuid4().hex, "token": os.environ["ALPHONSE_BRIDGE_TOKEN"], "tool_id": tool_id, "arguments": arguments, "parallel": parallel}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect("/bridge/tools.sock")
            client.sendall((json.dumps(request) + "\\n").encode())
            response = b""
            while not response.endswith(b"\\n"): response += client.recv(65536)
        return json.loads(response.decode())["result"]
'''
