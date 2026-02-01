from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from urllib import request

from dotenv import load_dotenv

from alphonse.agent.cognition.intentions.intent_pipeline import (
    build_default_pipeline_with_bus,
)
from alphonse.agent.heart import Heart, HeartConfig, SHUTDOWN
from alphonse.agent.nervous_system.ddfsm import DDFSM, DDFSMConfig
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.senses.timer import TimerSense
from alphonse.agent.nervous_system.seed import apply_seed
from alphonse.agent.nervous_system.capability_gaps import (
    get_gap,
    list_gaps,
    update_gap_status,
)
from alphonse.agent.core.settings_store import init_db as init_settings_db


def main() -> None:
    parser = argparse.ArgumentParser(prog="alphonse cli")
    parser.add_argument("--log-level", default=os.getenv("ALPHONSE_LOG_LEVEL", "INFO"))
    sub = parser.add_subparsers(dest="command", required=True)

    say_parser = sub.add_parser(
        "say", help="Send a message through the cortex pipeline"
    )
    say_parser.add_argument("text", help="Message text")
    say_parser.add_argument("--chat-id", default="cli", help="Channel target/chat id")
    say_parser.add_argument(
        "--channel", default="cli", help="Origin channel (cli, telegram, api)"
    )
    say_parser.add_argument("--person-id", default=None, help="Optional person id")
    say_parser.add_argument(
        "--correlation-id", default=None, help="Optional correlation id"
    )

    run_parser = sub.add_parser(
        "run-scheduler", help="Run the timed signal dispatcher loop"
    )
    run_parser.add_argument(
        "--poll-seconds", type=float, default=None, help="Override poll interval"
    )

    status_parser = sub.add_parser("status", help="Show timed signal status summary")

    report_parser = sub.add_parser("report", help="Report utilities")
    report_sub = report_parser.add_subparsers(dest="report_command", required=True)
    report_daily = report_sub.add_parser(
        "daily", help="Show daily report schedule and last sent"
    )

    debug_parser = sub.add_parser("debug", help="Diagnostics")
    debug_sub = debug_parser.add_subparsers(dest="debug_command", required=True)
    debug_wiring = debug_sub.add_parser(
        "wiring", help="Show DB path and timed signals API status"
    )

    gaps_parser = sub.add_parser("gaps", help="Inspect capability gaps")
    gaps_sub = gaps_parser.add_subparsers(dest="gaps_command", required=True)
    gaps_list = gaps_sub.add_parser("list", help="List capability gaps")
    gaps_list.add_argument("--all", action="store_true", help="Include non-open gaps")
    gaps_list.add_argument(
        "--status",
        choices=["open", "triaged", "resolved", "ignored"],
        default=None,
        help="Filter by status",
    )
    gaps_list.add_argument("--limit", type=int, default=50, help="Limit results")

    gaps_show = gaps_sub.add_parser("show", help="Show a gap by id")
    gaps_show.add_argument("gap_id", help="Gap id")

    gaps_resolve = gaps_sub.add_parser("resolve", help="Resolve a gap")
    gaps_resolve.add_argument("gap_id", help="Gap id")
    gaps_resolve.add_argument("--note", required=True, help="Resolution note")

    gaps_ignore = gaps_sub.add_parser("ignore", help="Ignore a gap")
    gaps_ignore.add_argument("gap_id", help="Gap id")
    gaps_ignore.add_argument("--note", required=True, help="Reason for ignoring")

    args = parser.parse_args()
    _configure_logging(args.log_level)
    _load_env()
    init_settings_db()
    db_path = resolve_nervous_system_db_path()
    logging.info("Nerve DB path=%s exists=%s", db_path, db_path.exists())
    apply_schema(db_path)
    apply_seed(db_path)

    if args.command == "say":
        _command_say(args, db_path)
        return
    if args.command == "run-scheduler":
        _command_run_scheduler(args, db_path)
        return
    if args.command == "status":
        _command_status(db_path)
        return
    if args.command == "report":
        _command_report(args, db_path)
        return
    if args.command == "debug":
        _command_debug(args)
        return
    if args.command == "gaps":
        _command_gaps(args)
        return


def _command_say(args: argparse.Namespace, db_path: Path) -> None:
    _ = db_path
    bus = Bus()
    pipeline = build_default_pipeline_with_bus(bus)
    correlation_id = args.correlation_id or str(uuid.uuid4())
    channel = str(args.channel).strip() or "cli"
    signal_type = (
        f"{channel}.message_received"
        if channel in {"telegram", "cli", "api"}
        else "cli.message_received"
    )
    payload = {
        "text": args.text,
        "chat_id": args.chat_id,
        "origin": channel,
        "timestamp": time.time(),
    }
    if args.person_id:
        payload["person_id"] = args.person_id
    signal = Signal(
        type=signal_type,
        payload=payload,
        source=channel,
        correlation_id=correlation_id,
    )
    pipeline.handle(
        "handle_incoming_message",
        {
            "signal": signal,
            "state": None,
            "outcome": None,
            "ctx": None,
        },
    )


def _command_run_scheduler(args: argparse.Namespace, db_path: Path) -> None:
    if args.poll_seconds is not None:
        os.environ["TIMER_POLL_SECONDS"] = str(args.poll_seconds)
    bus = Bus()
    ddfsm = DDFSM(DDFSMConfig(db_path=str(db_path)))
    heart = Heart(
        HeartConfig(nervous_system_db_path=str(db_path)), bus=bus, ddfsm=ddfsm
    )
    timer = TimerSense()
    timer.start(bus)
    logging.info("Scheduler loop started (ctrl+c to stop)")
    try:
        heart.run()
    except KeyboardInterrupt:
        bus.emit(Signal(type=SHUTDOWN, payload={}, source="cli"))
    finally:
        timer.stop()


def _command_status(db_path: Path) -> None:
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM timed_signals GROUP BY status"
        ).fetchall()
        due = conn.execute(
            """
            SELECT COUNT(*) FROM timed_signals
            WHERE status = 'pending' AND COALESCE(next_trigger_at, trigger_at) <= ?
            """,
            (now_iso,),
        ).fetchone()
    print("Timed signals by status:")
    for status, count in rows:
        print(f"- {status}: {count}")
    if due:
        print(f"Due now: {due[0]}")


def _command_report(args: argparse.Namespace, db_path: Path) -> None:
    if args.report_command == "daily":
        _command_report_daily(db_path)
        return


def _command_report_daily(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, signal_type, status, trigger_at, next_trigger_at, fired_at, rrule
            FROM timed_signals WHERE id = 'daily_report'
            """
        ).fetchone()
    if not row:
        print("Daily report schedule not found.")
        return
    print("Daily report schedule:")
    print(f"- id: {row[0]}")
    print(f"- signal_type: {row[1]}")
    print(f"- status: {row[2]}")
    print(f"- trigger_at: {row[3]}")
    print(f"- next_trigger_at: {row[4]}")
    print(f"- fired_at: {row[5]}")
    print(f"- rrule: {row[6]}")


def _command_debug(args: argparse.Namespace) -> None:
    if args.debug_command == "wiring":
        _command_debug_wiring()
        return


def _command_debug_wiring() -> None:
    db_path = resolve_nervous_system_db_path()
    api_base = os.getenv("ALPHONSE_API_BASE_URL", "http://localhost:8001").rstrip("/")
    token = os.getenv("ALPHONSE_API_TOKEN")
    print(f"DB path: {db_path}")
    print(f"API base: {api_base}")
    print("API token set: yes" if token else "API token set: no")

    url = f"{api_base}/agent/timed-signals"
    req = request.Request(url, method="GET")
    if token:
        req.add_header("x-alphonse-api-token", token)
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, dict) and "data" in data:
                timed_signals = data.get("data", {}).get("timed_signals", [])
            else:
                timed_signals = (
                    data.get("timed_signals", []) if isinstance(data, dict) else []
                )
            print(f"API timed signals count: {len(timed_signals)}")
    except Exception as exc:
        print(f"API timed signals error: {exc}")


def _command_gaps(args: argparse.Namespace) -> None:
    if args.gaps_command == "list":
        _command_gaps_list(args)
        return
    if args.gaps_command == "show":
        _command_gaps_show(args)
        return
    if args.gaps_command == "resolve":
        _command_gaps_update(args, status="resolved")
        return
    if args.gaps_command == "ignore":
        _command_gaps_update(args, status="ignored")
        return


def _command_gaps_list(args: argparse.Namespace) -> None:
    rows = list_gaps(
        status=args.status,
        limit=args.limit,
        include_all=args.all,
    )
    if not rows:
        print("No capability gaps found.")
        return
    for row in rows:
        print(
            f"{row['gap_id']} | {row['status']} | {row['reason']} | {row['created_at']} | {row['user_text']}"
        )


def _command_gaps_show(args: argparse.Namespace) -> None:
    row = get_gap(args.gap_id)
    if not row:
        print("Gap not found.")
        return
    for key, value in row.items():
        print(f"{key}: {value}")


def _command_gaps_update(args: argparse.Namespace, *, status: str) -> None:
    updated = update_gap_status(args.gap_id, status=status, note=args.note)
    if updated:
        print(f"Updated {args.gap_id} to {status}.")
    else:
        print("Gap not found.")


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, level_name.upper(), logging.INFO))


if __name__ == "__main__":
    main()
