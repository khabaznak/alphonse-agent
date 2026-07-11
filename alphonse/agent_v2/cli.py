"""Command-line lifecycle controls for the v2 daemon."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from alphonse.agent_v2.ipc import V2DaemonClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphonse", description="Alphonse v2 daemon controls")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start", help="Start the v2 daemon or TUI")
    start_parser.add_subparsers(dest="start_target", required=False).add_parser(
        "tui", help="Start the daemon if needed, then open the TUI"
    )
    subparsers.add_parser("status", help="Show v2 daemon health and queue status")
    subparsers.add_parser("stop", help="Gracefully stop the v2 daemon")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "start":
        if getattr(args, "start_target", "") == "tui":
            return start_tui()
        return start_daemon()
    if args.command == "status":
        return show_status()
    if args.command == "stop":
        return stop_daemon()
    return 2


def start_daemon(*, client: V2DaemonClient | None = None) -> int:
    client = client or V2DaemonClient(timeout_sec=0.5)
    try:
        client.ping()
    except Exception:
        log_path = _daemon_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        try:
            subprocess.Popen(
                [sys.executable, "-m", "alphonse.agent_v2.daemon"],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=os.environ.copy(),
            )
        finally:
            log_file.close()
        if not _wait_for_daemon(client):
            print(f"Alphonse daemon did not become ready. Check {log_path}", file=sys.stderr)
            return 1
        print("Alphonse daemon started.")
        return 0
    print("Alphonse daemon is already running.")
    return 0


def show_status(*, client: V2DaemonClient | None = None) -> int:
    client = client or V2DaemonClient(timeout_sec=2.0)
    try:
        status = client.status()
    except Exception as exc:
        print(format_unavailable_status(exc))
        return 1
    print(format_status(status))
    return 0


def stop_daemon(*, client: V2DaemonClient | None = None) -> int:
    client = client or V2DaemonClient(timeout_sec=0.75)
    try:
        client.stop()
    except Exception:
        print("Alphonse daemon is not running.")
        return 1
    if not _wait_for_shutdown(client):
        print("Alphonse daemon is still stopping. Check the daemon log.", file=sys.stderr)
        return 1
    print("Alphonse daemon stopped.")
    return 0


def start_tui() -> int:
    if start_daemon() != 0:
        return 1
    completed = subprocess.run(
        [sys.executable, "-m", "alphonse.agent_v2.tui"],
        env=os.environ.copy(),
        check=False,
    )
    return int(completed.returncode)


def format_status(status: dict[str, Any]) -> str:
    scheduler = status.get("scheduler") if isinstance(status.get("scheduler"), dict) else {}
    inbound = status.get("inbound_counts") if isinstance(status.get("inbound_counts"), dict) else {}
    outbound = status.get("outbound_counts") if isinstance(status.get("outbound_counts"), dict) else {}
    lines = [
        "Alphonse daemon: running",
        f"daemon id: {status.get('daemon_id') or '-'}",
        (
            "inbound queue: "
            f"{status.get('queue_size', 0)} ready, {inbound.get('processing', 0)} processing, "
            f"{inbound.get('retry_wait', 0)} retrying, {inbound.get('failed', 0)} failed"
        ),
        (
            "outbound queue: "
            f"{status.get('outbound_queue_size', 0)} ready, {outbound.get('claimed', 0)} delivering, "
            f"{outbound.get('retry_wait', 0)} retrying"
        ),
        f"due schedules: {status.get('due_schedules', 0)}",
        f"processor: {'alive' if status.get('processor_alive') else 'stopped'}",
        f"scheduler ticks: {scheduler.get('ticks', 0)}",
        f"scheduled claimed: {scheduler.get('claimed', 0)}",
        f"scheduled queued: {scheduler.get('queued', 0)}",
        f"scheduled retries: {scheduler.get('retried', 0)}",
    ]
    active_work = format_active_work(status.get("active_work"))
    if active_work:
        lines.append(f"active work: {active_work}")
    if status.get("last_processor_error"):
        lines.append(f"processor error: {status['last_processor_error']}")
    if status.get("status_error"):
        lines.append(f"status warning: {status['status_error']}")
    return "\n".join(lines)


def format_active_work(active_work: Any) -> str:
    if not isinstance(active_work, dict):
        return ""
    prompt = " ".join(str(active_work.get("prompt") or "").split())
    if not prompt:
        return ""
    user = str(active_work.get("user") or "unknown")
    return f"{user}: {prompt[:120]}"


def format_unavailable_status(error: Exception) -> str:
    """Turn expected local IPC connection failures into lifecycle guidance."""
    if isinstance(error, (FileNotFoundError, ConnectionRefusedError, TimeoutError)):
        return "Alphonse daemon: stopped. Run 'alphonse start' to launch it."
    return f"Alphonse daemon status unavailable: {type(error).__name__}: {error}"


def _wait_for_daemon(client: V2DaemonClient, *, timeout_sec: float = 5.0) -> bool:
    deadline = time.monotonic() + max(0.5, timeout_sec)
    while time.monotonic() < deadline:
        try:
            client.ping()
            return True
        except Exception:
            time.sleep(0.1)
    return False


def _wait_for_shutdown(client: V2DaemonClient, *, timeout_sec: float = 5.0) -> bool:
    deadline = time.monotonic() + max(0.5, timeout_sec)
    while time.monotonic() < deadline:
        try:
            client.ping()
        except Exception:
            return True
        time.sleep(0.1)
    return False


def _daemon_log_path() -> Path:
    return Path(os.getenv("ALPHONSE_V2_DAEMON_LOG_PATH") or Path.home() / ".alphonse" / "v2-daemon.log")


if __name__ == "__main__":
    raise SystemExit(main())
