from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

from alphonse.agent.cognition.intentions.intent_pipeline import build_default_pipeline_with_bus
from alphonse.agent.heart import Heart, HeartConfig, SHUTDOWN
from alphonse.agent.nervous_system.ddfsm import DDFSM, DDFSMConfig
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.senses.timer import TimerSense
from alphonse.agent.nervous_system.seed import apply_seed
from alphonse.agent.core.settings_store import init_db as init_settings_db


def main() -> None:
    parser = argparse.ArgumentParser(prog="alphonse cli")
    parser.add_argument("--log-level", default=os.getenv("ALPHONSE_LOG_LEVEL", "INFO"))
    sub = parser.add_subparsers(dest="command", required=True)

    say_parser = sub.add_parser("say", help="Send a message through the cortex pipeline")
    say_parser.add_argument("text", help="Message text")
    say_parser.add_argument("--chat-id", default="cli", help="Channel target/chat id")
    say_parser.add_argument("--channel", default="cli", help="Origin channel (cli, telegram, api)")
    say_parser.add_argument("--person-id", default=None, help="Optional person id")
    say_parser.add_argument("--correlation-id", default=None, help="Optional correlation id")

    run_parser = sub.add_parser("run-scheduler", help="Run the timed signal dispatcher loop")
    run_parser.add_argument("--poll-seconds", type=float, default=None, help="Override poll interval")

    status_parser = sub.add_parser("status", help="Show timed signal status summary")

    args = parser.parse_args()
    _configure_logging(args.log_level)
    _load_env()
    init_settings_db()
    db_path = resolve_nervous_system_db_path()
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


def _command_say(args: argparse.Namespace, db_path: Path) -> None:
    _ = db_path
    bus = Bus()
    pipeline = build_default_pipeline_with_bus(bus)
    correlation_id = args.correlation_id or str(uuid.uuid4())
    channel = str(args.channel).strip() or "cli"
    signal_type = f"{channel}.message_received" if channel in {"telegram", "cli", "api"} else "cli.message_received"
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
    heart = Heart(HeartConfig(nervous_system_db_path=str(db_path)), bus=bus, ddfsm=ddfsm)
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


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, level_name.upper(), logging.INFO))


if __name__ == "__main__":
    main()
