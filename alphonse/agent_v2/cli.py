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
    subparsers.add_parser("start", help="Start the v2 daemon in the background")
    subparsers.add_parser("status", help="Show v2 daemon health and queue status")
    subparsers.add_parser("stop", help="Gracefully stop the v2 daemon")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "start":
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
    client = client or V2DaemonClient(timeout_sec=0.75)
    try:
        status = client.status()
    except Exception:
        print("Alphonse daemon: stopped")
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
    print("Alphonse daemon stopping.")
    return 0


def format_status(status: dict[str, Any]) -> str:
    scheduler = status.get("scheduler") if isinstance(status.get("scheduler"), dict) else {}
    lines = [
        "Alphonse daemon: running",
        f"daemon id: {status.get('daemon_id') or '-'}",
        f"inbound queue: {status.get('queue_size', 0)}",
        f"outbound queue: {status.get('outbound_queue_size', 0)}",
        f"due schedules: {status.get('due_schedules', 0)}",
        f"scheduler ticks: {scheduler.get('ticks', 0)}",
        f"scheduled claimed: {scheduler.get('claimed', 0)}",
        f"scheduled queued: {scheduler.get('queued', 0)}",
        f"scheduled retries: {scheduler.get('retried', 0)}",
    ]
    return "\n".join(lines)


def _wait_for_daemon(client: V2DaemonClient, *, timeout_sec: float = 5.0) -> bool:
    deadline = time.monotonic() + max(0.5, timeout_sec)
    while time.monotonic() < deadline:
        try:
            client.ping()
            return True
        except Exception:
            time.sleep(0.1)
    return False


def _daemon_log_path() -> Path:
    return Path(os.getenv("ALPHONSE_V2_DAEMON_LOG_PATH") or Path.home() / ".alphonse" / "v2-daemon.log")


if __name__ == "__main__":
    raise SystemExit(main())
