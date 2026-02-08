from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
import uuid
import shlex
import signal
import subprocess
import sys
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
from alphonse.agent.cognition.capability_gaps.reflection import reflect_gaps
from alphonse.agent.cognition.abilities.store import AbilitySpecStore
from alphonse.agent.nervous_system.gap_proposals import (
    get_gap_proposal,
    list_gap_proposals,
    update_gap_proposal_status,
)
from alphonse.agent.nervous_system.gap_tasks import (
    get_gap_task,
    insert_gap_task,
    list_gap_tasks,
    update_gap_task_status,
)
from alphonse.agent.nervous_system.onboarding_profiles import (
    delete_onboarding_profile,
    get_onboarding_profile,
    list_onboarding_profiles,
    upsert_onboarding_profile,
)
from alphonse.agent.nervous_system.location_profiles import (
    delete_location_profile,
    get_location_profile,
    insert_device_location,
    list_device_locations,
    list_location_profiles,
    upsert_location_profile,
)
from alphonse.agent.core.settings_store import init_db as init_settings_db


def build_parser() -> argparse.ArgumentParser:
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
    say_parser.add_argument(
        "--planning-mode",
        choices=["aventurizacion", "contrato_de_resultado"],
        default=None,
        help="Override planning mode for this message",
    )
    say_parser.add_argument(
        "--autonomy-level",
        type=float,
        default=None,
        help="Override autonomy level (0.0 - 1.0)",
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
    debug_intent = debug_sub.add_parser(
        "intent", help="Inspect intent routing for a message"
    )
    debug_intent.add_argument("text", nargs="+", help="Message text to route")

    agent_parser = sub.add_parser("agent", help="Manage agent process (REPL-only)")
    agent_sub = agent_parser.add_subparsers(dest="agent_command", required=True)
    agent_sub.add_parser("start", help="Start managed agent process")
    agent_sub.add_parser("stop", help="Stop managed agent process")
    agent_sub.add_parser("restart", help="Restart managed agent process")
    agent_sub.add_parser("status", help="Show managed agent status")

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

    gaps_reflect = gaps_sub.add_parser("reflect", help="Propose actions for open gaps")
    gaps_reflect.add_argument("--limit", type=int, default=50, help="Limit open gaps")

    proposals_parser = gaps_sub.add_parser("proposals", help="Manage gap proposals")
    proposals_sub = proposals_parser.add_subparsers(
        dest="proposals_command", required=True
    )
    proposals_list = proposals_sub.add_parser("list", help="List proposals")
    proposals_list.add_argument(
        "--status",
        choices=["pending", "approved", "rejected", "dispatched"],
        default="pending",
    )
    proposals_list.add_argument("--limit", type=int, default=50, help="Limit results")
    proposals_show = proposals_sub.add_parser("show", help="Show proposal details")
    proposals_show.add_argument("proposal_id", help="Proposal id")
    proposals_approve = proposals_sub.add_parser("approve", help="Approve proposal")
    proposals_approve.add_argument("proposal_id", help="Proposal id")
    proposals_approve.add_argument(
        "--queue", choices=["plan", "investigate", "fix_now"], default=None
    )
    proposals_approve.add_argument("--reviewer", default="human")
    proposals_reject = proposals_sub.add_parser("reject", help="Reject proposal")
    proposals_reject.add_argument("proposal_id", help="Proposal id")
    proposals_reject.add_argument("--reason", default=None)

    tasks_parser = gaps_sub.add_parser("tasks", help="Manage gap tasks")
    tasks_sub = tasks_parser.add_subparsers(dest="tasks_command", required=True)
    tasks_list = tasks_sub.add_parser("list", help="List tasks")
    tasks_list.add_argument("--status", choices=["open", "done"], default="open")
    tasks_list.add_argument("--limit", type=int, default=50)
    tasks_show = tasks_sub.add_parser("show", help="Show task details")
    tasks_show.add_argument("task_id", help="Task id")
    tasks_done = tasks_sub.add_parser("done", help="Mark task as done")
    tasks_done.add_argument("task_id", help="Task id")

    lan_parser = sub.add_parser("lan", help="LAN pairing utilities")
    lan_sub = lan_parser.add_subparsers(dest="lan_command", required=True)
    lan_code = lan_sub.add_parser("pairing-code", help="Generate a pairing code")
    lan_code.add_argument(
        "--ttl-minutes",
        type=int,
        default=None,
        help="Override pairing code TTL minutes",
    )
    lan_devices = lan_sub.add_parser("devices", help="List paired devices")
    lan_devices.add_argument("--limit", type=int, default=25)
    lan_token = lan_sub.add_parser("relay-token", help="Mint relay access token")
    lan_token.add_argument("device_id", help="Device id")
    lan_token.add_argument("--device-name", default=None, help="Optional device name")

    repl_parser = sub.add_parser("repl", help="Start interactive CLI session")

    catalog_parser = sub.add_parser("catalog", help="Intent catalog admin")
    catalog_sub = catalog_parser.add_subparsers(dest="catalog_command", required=True)
    catalog_list = catalog_sub.add_parser("list", help="List catalog intents")
    catalog_list.add_argument(
        "--all", action="store_true", help="Include disabled intents"
    )
    catalog_show = catalog_sub.add_parser("show", help="Show an intent spec")
    catalog_show.add_argument("intent_name", help="Intent name")
    catalog_enable = catalog_sub.add_parser("enable", help="Enable an intent")
    catalog_enable.add_argument("intent_name", help="Intent name")
    catalog_disable = catalog_sub.add_parser("disable", help="Disable an intent")
    catalog_disable.add_argument("intent_name", help="Intent name")
    catalog_sub.add_parser("refresh", help="Refresh catalog cache")
    catalog_sub.add_parser("seed", help="Seed factory intents into the catalog")
    catalog_sub.add_parser("validate", help="Validate catalog entries")
    catalog_sub.add_parser("stats", help="Show catalog diagnostics")
    catalog_prompt = catalog_sub.add_parser("prompt", help="Render routing prompt")
    catalog_prompt.add_argument("--text", required=True, help="User message to render")
    catalog_prompt.add_argument(
        "--mode",
        choices=("map", "legacy"),
        default="map",
        help="Prompt mode: map (default runtime) or legacy detector",
    )
    catalog_template = catalog_sub.add_parser("template", help="Show prompt template by key")
    catalog_template.add_argument("key", help="Prompt template key")

    abilities_parser = sub.add_parser("abilities", help="Abilities CRUD")
    abilities_sub = abilities_parser.add_subparsers(
        dest="abilities_command", required=True
    )
    abilities_list = abilities_sub.add_parser("list", help="List ability specs")
    abilities_list.add_argument(
        "--enabled-only",
        action="store_true",
        help="Show only enabled ability specs",
    )
    abilities_list.add_argument("--limit", type=int, default=100, help="Limit results")
    abilities_show = abilities_sub.add_parser("show", help="Show ability spec by intent")
    abilities_show.add_argument("intent_name", help="Intent name")
    abilities_create = abilities_sub.add_parser("create", help="Create ability spec")
    abilities_create.add_argument("intent_name", help="Intent name")
    abilities_create.add_argument("kind", help="Ability kind")
    abilities_create.add_argument(
        "--tools",
        nargs="*",
        default=[],
        help="Tool names (space-separated)",
    )
    abilities_create.add_argument(
        "--spec-json",
        default=None,
        help="Ability spec JSON object override",
    )
    abilities_create.add_argument(
        "--spec-file",
        default=None,
        help="Path to ability spec JSON file",
    )
    abilities_create.add_argument("--source", default="user", help="Ability source")
    abilities_create.add_argument(
        "--disabled",
        action="store_true",
        help="Create with enabled=false",
    )
    abilities_update = abilities_sub.add_parser("update", help="Update ability spec")
    abilities_update.add_argument("intent_name", help="Intent name")
    abilities_update.add_argument("--kind", default=None, help="New kind")
    abilities_update.add_argument(
        "--tools",
        nargs="*",
        default=None,
        help="Replacement tool names",
    )
    abilities_update.add_argument(
        "--spec-json",
        default=None,
        help="Spec JSON patch object",
    )
    abilities_update.add_argument(
        "--spec-file",
        default=None,
        help="Path to spec JSON patch file",
    )
    abilities_update.add_argument("--source", default=None, help="Updated source")
    abilities_enable = abilities_sub.add_parser("enable", help="Enable ability spec")
    abilities_enable.add_argument("intent_name", help="Intent name")
    abilities_disable = abilities_sub.add_parser("disable", help="Disable ability spec")
    abilities_disable.add_argument("intent_name", help="Intent name")
    abilities_delete = abilities_sub.add_parser("delete", help="Delete ability spec")
    abilities_delete.add_argument("intent_name", help="Intent name")

    onboarding_parser = sub.add_parser("onboarding", help="Onboarding profile CRUD")
    onboarding_sub = onboarding_parser.add_subparsers(
        dest="onboarding_command", required=True
    )
    onboarding_list = onboarding_sub.add_parser("list", help="List onboarding profiles")
    onboarding_list.add_argument("--state", default=None)
    onboarding_list.add_argument("--limit", type=int, default=100)
    onboarding_show = onboarding_sub.add_parser("show", help="Show onboarding profile")
    onboarding_show.add_argument("principal_id")
    onboarding_upsert = onboarding_sub.add_parser("upsert", help="Create/update onboarding profile")
    onboarding_upsert.add_argument("principal_id")
    onboarding_upsert.add_argument("--state", default="not_started")
    onboarding_upsert.add_argument("--primary-role", default=None)
    onboarding_upsert.add_argument("--next-steps", nargs="*", default=[])
    onboarding_upsert.add_argument("--resume-token", default=None)
    onboarding_upsert.add_argument("--completed-at", default=None)
    onboarding_delete = onboarding_sub.add_parser("delete", help="Delete onboarding profile")
    onboarding_delete.add_argument("principal_id")

    locations_parser = sub.add_parser("locations", help="Location profile CRUD")
    locations_sub = locations_parser.add_subparsers(
        dest="locations_command", required=True
    )
    locations_list = locations_sub.add_parser("list", help="List location profiles")
    locations_list.add_argument("--principal-id", default=None)
    locations_list.add_argument("--label", default=None)
    locations_list.add_argument("--active-only", action="store_true")
    locations_list.add_argument("--limit", type=int, default=100)
    locations_show = locations_sub.add_parser("show", help="Show location profile")
    locations_show.add_argument("location_id")
    locations_upsert = locations_sub.add_parser("upsert", help="Create/update location profile")
    locations_upsert.add_argument("--location-id", default=None)
    locations_upsert.add_argument("principal_id")
    locations_upsert.add_argument("--label", default="other")
    locations_upsert.add_argument("--address-text", default=None)
    locations_upsert.add_argument("--lat", type=float, default=None)
    locations_upsert.add_argument("--lng", type=float, default=None)
    locations_upsert.add_argument("--source", default="user")
    locations_upsert.add_argument("--confidence", type=float, default=None)
    locations_upsert.add_argument("--inactive", action="store_true")
    locations_delete = locations_sub.add_parser("delete", help="Delete location profile")
    locations_delete.add_argument("location_id")
    device_add = locations_sub.add_parser("device-add", help="Insert device location sample")
    device_add.add_argument("device_id")
    device_add.add_argument("--principal-id", default=None)
    device_add.add_argument("--lat", type=float, required=True)
    device_add.add_argument("--lng", type=float, required=True)
    device_add.add_argument("--accuracy", type=float, default=None)
    device_add.add_argument("--source", default="device")
    device_add.add_argument("--observed-at", default=None)
    device_add.add_argument("--metadata-json", default=None)
    device_list = locations_sub.add_parser("device-list", help="List device locations")
    device_list.add_argument("--principal-id", default=None)
    device_list.add_argument("--device-id", default=None)
    device_list.add_argument("--limit", type=int, default=100)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)
    _load_env()
    init_settings_db()
    db_path = resolve_nervous_system_db_path()
    logging.info("Nerve DB path=%s exists=%s", db_path, db_path.exists())
    apply_schema(db_path)
    apply_seed(db_path)

    _dispatch_command(args, db_path, parser)
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
    if args.planning_mode:
        payload["planning_mode"] = args.planning_mode
    if args.autonomy_level is not None:
        payload["autonomy_level"] = args.autonomy_level
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


def _dispatch_command(
    args: argparse.Namespace,
    db_path: Path,
    parser: argparse.ArgumentParser,
    *,
    supervisor: "AgentSupervisor | None" = None,
) -> None:
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
    if args.command == "agent":
        _command_agent(args, supervisor=supervisor)
        return
    if args.command == "gaps":
        _command_gaps(args)
        return
    if args.command == "lan":
        _command_lan(args)
        return
    if args.command == "catalog":
        _command_catalog(args)
        return
    if args.command == "abilities":
        _command_abilities(args)
        return
    if args.command == "onboarding":
        _command_onboarding(args)
        return
    if args.command == "locations":
        _command_locations(args)
        return
    if args.command == "repl":
        _command_repl(parser, db_path)
        return


def _command_repl(parser: argparse.ArgumentParser, db_path: Path) -> None:
    supervisor = AgentSupervisor()
    while True:
        try:
            raw = input("alphonse> ").strip()
        except EOFError:
            break
        if not raw:
            continue
        if raw in {"exit", "quit"}:
            break
        if raw in {"help", "?"}:
            parser.print_help()
            continue
        try:
            args = parser.parse_args(shlex.split(raw))
        except SystemExit:
            continue
        _configure_logging(args.log_level)
        if args.command == "repl":
            print("Already in repl. Type a command or 'exit'.")
            continue
        _dispatch_command(args, db_path, parser, supervisor=supervisor)
    if supervisor.is_running():
        print("Managed agent is still running. Use 'agent stop' to stop it.")


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


def _command_lan(args: argparse.Namespace) -> None:
    from alphonse.agent.lan.store import generate_pairing_code, list_paired_devices
    from alphonse.agent.lan.qr import render_ascii_qr
    from alphonse.agent.relay.issuer import mint_relay_token

    if args.lan_command == "pairing-code":
        ttl = args.ttl_minutes if args.ttl_minutes is not None else None
        code = generate_pairing_code(ttl_minutes=ttl or 15)
        print("Pairing code:")
        print(f"- code: {code.code}")
        print(f"- expires_at: {code.expires_at.isoformat()}")
        print("\nQR:\n")
        print(render_ascii_qr(code.code))
        return

    if args.lan_command == "devices":
        devices = list_paired_devices(limit=args.limit)
        if not devices:
            print("No paired devices.")
            return
        print("Paired devices:")
        for device in devices:
            name = device.device_name or "Unnamed device"
            last_seen = device.last_seen_at.isoformat() if device.last_seen_at else "never"
            print(f"- {device.device_id} ({name}) last_seen={last_seen}")
        return

    if args.lan_command == "relay-token":
        result = mint_relay_token(args.device_id, args.device_name)
        if not result:
            print("Relay not configured or channel creation failed.")
            return
        print(result.get("relay_token"))
        return
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


def _command_catalog(args: argparse.Namespace) -> None:
    from alphonse.agent.cognition.intent_catalog import (
        IntentCatalogStore,
        get_catalog_service,
        seed_default_intents,
    )

    store = IntentCatalogStore()
    if args.catalog_command == "list":
        if args.all:
            rows = store.list_all()
        else:
            rows = store.list_enabled()
        if not rows:
            print("No intents found.")
            return
        print("Intent catalog:")
        for spec in rows:
            status = "enabled" if spec.enabled else "disabled"
            print(f"- {spec.intent_name} ({spec.category}) {status}")
        return

    if args.catalog_command == "show":
        spec = store.get(args.intent_name)
        if not spec:
            print(f"Intent not found: {args.intent_name}")
            return
        print(f"Intent: {spec.intent_name}")
        print(f"- category: {spec.category}")
        print(f"- enabled: {spec.enabled}")
        print(f"- description: {spec.description}")
        print(f"- default_mode: {spec.default_mode}")
        print(f"- risk_level: {spec.risk_level}")
        print(f"- handler: {spec.handler}")
        if spec.examples:
            print("- examples:")
            for example in spec.examples:
                print(f"  - {example}")
        if spec.required_slots:
            print("- required_slots:")
            for slot in spec.required_slots:
                print(
                    f"  - {slot.name} ({slot.type}) required={slot.required} critical={slot.critical}"
                )
        if spec.optional_slots:
            print("- optional_slots:")
            for slot in spec.optional_slots:
                print(
                    f"  - {slot.name} ({slot.type}) required={slot.required} critical={slot.critical}"
                )
        return

    if args.catalog_command in {"enable", "disable"}:
        enabled = args.catalog_command == "enable"
        spec = store.get(args.intent_name)
        if not spec:
            print(f"Intent not found: {args.intent_name}")
            return
        updated = type(spec)(
            intent_name=spec.intent_name,
            category=spec.category,
            description=spec.description,
            examples=spec.examples,
            required_slots=spec.required_slots,
            optional_slots=spec.optional_slots,
            default_mode=spec.default_mode,
            risk_level=spec.risk_level,
            handler=spec.handler,
            enabled=enabled,
        )
        store.upsert(updated)
        state = "enabled" if enabled else "disabled"
        print(f"{spec.intent_name} -> {state}")
        return

    if args.catalog_command == "refresh":
        service = get_catalog_service()
        intents = service.refresh()
        diag = service.diagnostics()
        print(f"Catalog refreshed: enabled={len(intents)} total={diag.total_count}")
        return

    if args.catalog_command == "seed":
        seed_default_intents()
        print("Catalog seeded.")
        return

    if args.catalog_command == "validate":
        errors = _validate_catalog(store)
        if not errors:
            print("Catalog validation passed.")
            return
        print("Catalog validation errors:")
        for err in errors:
            print(f"- {err}")
        return

    if args.catalog_command == "stats":
        service = get_catalog_service()
        diag = service.diagnostics()
        print("Catalog diagnostics:")
        print(f"- db_path: {diag.db_path}")
        print(f"- available: {diag.available}")
        print(f"- enabled_count: {diag.enabled_count}")
        print(f"- total_count: {diag.total_count}")
        print(f"- last_refresh_at: {diag.last_refresh_at}")
        if diag.categories:
            print("- categories:")
            for key, value in sorted(diag.categories.items()):
                print(f"  - {key}: {value}")
        return

    if args.catalog_command == "prompt":
        if args.mode == "legacy":
            from alphonse.agent.cognition.intent_detector_llm import build_detector_prompt

            service = get_catalog_service()
            prompt = build_detector_prompt(args.text, service)
            if prompt is None:
                print("No enabled intents; prompt not available.")
                return
            print("Mode: legacy")
            print(prompt)
            return
        from alphonse.agent.cognition.message_map_llm import build_message_map_prompt

        prompt = build_message_map_prompt(args.text)
        print("Mode: map")
        print(prompt)
        return

    if args.catalog_command == "template":
        from alphonse.agent.cognition.prompt_store import SqlitePromptStore, PromptContext

        store = SqlitePromptStore()
        match = store.get_template(
            args.key,
            PromptContext(
                locale="any",
                address_style="any",
                tone="any",
                channel="any",
                variant="default",
                policy_tier="safe",
            ),
        )
        if not match:
            print(f"Template not found: {args.key}")
            return
        print(match.template)
        return


def _validate_catalog(store: IntentCatalogStore) -> list[str]:
    errors: list[str] = []
    try:
        specs = store.list_all()
    except Exception as exc:
        return [f"catalog unavailable: {exc}"]
    for spec in specs:
        if not spec.intent_name:
            errors.append("intent_name missing")
        if not spec.category:
            errors.append(f"{spec.intent_name}: category missing")
        if not spec.handler:
            errors.append(f"{spec.intent_name}: handler missing")
        if not spec.intent_version:
            errors.append(f"{spec.intent_name}: intent_version missing")
        if not spec.origin:
            errors.append(f"{spec.intent_name}: origin missing")
        for slot in spec.required_slots + spec.optional_slots:
            if not slot.name:
                errors.append(f"{spec.intent_name}: slot name missing")
            if not slot.type:
                errors.append(f"{spec.intent_name}: slot type missing")
            if not slot.prompt_key:
                errors.append(f"{spec.intent_name}: slot prompt_key missing")
    return errors


def _command_debug(args: argparse.Namespace) -> None:
    if args.debug_command == "wiring":
        _command_debug_wiring()
        return
    if args.debug_command == "intent":
        _command_debug_intent(args)
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


def _command_debug_intent(args: argparse.Namespace) -> None:
    from alphonse.agent.cognition.intent_catalog import get_catalog_service
    from alphonse.agent.cognition.intent_detector_llm import IntentDetectorLLM

    text = " ".join(args.text)
    service = get_catalog_service()
    detector = IntentDetectorLLM(service)
    detection = detector.detect(text, llm_client=None)
    print("Intent routing (catalog):")
    if not detection:
        print("- intent: unknown")
        print("- category: task_plane")
        print("- confidence: 0.00")
        print("- rationale: catalog_unavailable_or_no_match")
        print("- needs_clarification: True")
        return
    spec = service.get_intent(detection.intent_name)
    category = spec.category if spec else "task_plane"
    print(f"- intent: {detection.intent_name}")
    print(f"- category: {category}")
    print(f"- confidence: {detection.confidence:.2f}")
    print("- rationale: catalog_detector")
    print(f"- needs_clarification: {detection.needs_clarification}")


def _command_agent(args: argparse.Namespace, *, supervisor: "AgentSupervisor | None") -> None:
    if supervisor is None:
        print("Agent management is available in the REPL session only.")
        return
    if args.agent_command == "start":
        supervisor.start()
        return
    if args.agent_command == "stop":
        supervisor.stop()
        return
    if args.agent_command == "restart":
        supervisor.restart()
        return
    if args.agent_command == "status":
        supervisor.status()
        return


class AgentSupervisor:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.is_running():
            print(f"Agent already running (pid={self._process.pid}).")
            return
        cmd = [sys.executable, "-m", "alphonse.agent.main"]
        self._process = subprocess.Popen(cmd)
        print(f"Agent started (pid={self._process.pid}).")

    def stop(self) -> None:
        if not self.is_running():
            print("No managed agent process is running.")
            return
        assert self._process is not None
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        print("Agent stopped.")
        self._process = None

    def restart(self) -> None:
        if self.is_running():
            self.stop()
        self.start()

    def status(self) -> None:
        if not self.is_running():
            print("Agent status: stopped.")
            return
        assert self._process is not None
        print(f"Agent status: running (pid={self._process.pid}).")

    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None


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
    if args.gaps_command == "reflect":
        _command_gaps_reflect(args)
        return
    if args.gaps_command == "proposals":
        _command_gap_proposals(args)
        return
    if args.gaps_command == "tasks":
        _command_gap_tasks(args)
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


def _command_gaps_reflect(args: argparse.Namespace) -> None:
    created = reflect_gaps(limit=args.limit)
    if not created:
        print("No proposals created.")
        return
    print(f"Created {len(created)} proposal(s).")


def _command_gap_proposals(args: argparse.Namespace) -> None:
    if args.proposals_command == "list":
        rows = list_gap_proposals(status=args.status, limit=args.limit)
        if not rows:
            print("No proposals found.")
            return
        for row in rows:
            print(
                f"- {row.get('id')} gap={row.get('gap_id')} "
                f"{row.get('proposed_category')} "
                f"conf={row.get('confidence')} "
                f"next={row.get('proposed_next_action')}"
            )
        return
    if args.proposals_command == "show":
        proposal = get_gap_proposal(args.proposal_id)
        if not proposal:
            print("Proposal not found.")
            return
        print(json.dumps(proposal, indent=2, ensure_ascii=True))
        return
    if args.proposals_command == "approve":
        if update_gap_proposal_status(
            args.proposal_id, "approved", reviewer=args.reviewer
        ):
            print(f"Proposal {args.proposal_id} approved.")
            if args.queue:
                task_id = insert_gap_task(
                    {
                        "proposal_id": args.proposal_id,
                        "type": args.queue,
                        "status": "open",
                        "payload": {},
                    }
                )
                print(f"Queued task {task_id}.")
        else:
            print("Proposal not found.")
        return
    if args.proposals_command == "reject":
        if update_gap_proposal_status(
            args.proposal_id, "rejected", reviewer="human", notes=args.reason
        ):
            print(f"Proposal {args.proposal_id} rejected.")
        else:
            print("Proposal not found.")
        return


def _command_gap_tasks(args: argparse.Namespace) -> None:
    if args.tasks_command == "list":
        rows = list_gap_tasks(status=args.status, limit=args.limit)
        if not rows:
            print("No tasks found.")
            return
        for row in rows:
            print(
                f"- {row.get('id')} proposal={row.get('proposal_id')} "
                f"type={row.get('type')} status={row.get('status')}"
            )
        return
    if args.tasks_command == "show":
        task = get_gap_task(args.task_id)
        if not task:
            print("Task not found.")
            return
        print(json.dumps(task, indent=2, ensure_ascii=True))
        return
    if args.tasks_command == "done":
        if update_gap_task_status(args.task_id, "done"):
            print(f"Task {args.task_id} marked done.")
        else:
            print("Task not found.")


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _command_abilities(args: argparse.Namespace) -> None:
    store = AbilitySpecStore()
    if args.abilities_command == "list":
        rows = store.list_specs(enabled_only=args.enabled_only, limit=args.limit)
        if not rows:
            print("No ability specs found.")
            return
        for row in rows:
            state = "enabled" if row.get("enabled") else "disabled"
            print(
                f"- {row.get('intent_name')} kind={row.get('kind')} "
                f"tools={','.join(row.get('tools') or [])} {state}"
            )
        return

    if args.abilities_command == "show":
        item = store.get_spec(args.intent_name)
        if not item:
            print("Ability spec not found.")
            return
        print(json.dumps(item, indent=2, ensure_ascii=True))
        return

    if args.abilities_command == "create":
        try:
            spec = _load_spec_patch(args.spec_json, args.spec_file)
        except ValueError as exc:
            print(str(exc))
            return
        spec["intent_name"] = args.intent_name
        spec["kind"] = args.kind
        spec["tools"] = list(args.tools)
        store.upsert_spec(
            args.intent_name,
            spec,
            enabled=not args.disabled,
            source=args.source,
        )
        print(f"Created ability spec: {args.intent_name}")
        return

    if args.abilities_command == "update":
        current = store.get_spec(args.intent_name)
        if not current:
            print("Ability spec not found.")
            return
        current_spec = current.get("spec")
        if not isinstance(current_spec, dict):
            current_spec = {}
        spec = dict(current_spec)
        try:
            patch_spec = _load_spec_patch(args.spec_json, args.spec_file)
        except ValueError as exc:
            print(str(exc))
            return
        spec.update(patch_spec)
        if args.kind is not None:
            spec["kind"] = args.kind
        if args.tools is not None:
            spec["tools"] = list(args.tools)
        spec["intent_name"] = args.intent_name
        enabled = bool(current.get("enabled"))
        source = str(current.get("source") or "user")
        if args.source is not None:
            source = args.source
        store.upsert_spec(
            args.intent_name,
            spec,
            enabled=enabled,
            source=source,
        )
        print(f"Updated ability spec: {args.intent_name}")
        return

    if args.abilities_command in {"enable", "disable"}:
        current = store.get_spec(args.intent_name)
        if not current:
            print("Ability spec not found.")
            return
        spec = current.get("spec")
        if not isinstance(spec, dict):
            print("Ability spec invalid.")
            return
        enabled = args.abilities_command == "enable"
        store.upsert_spec(
            args.intent_name,
            spec,
            enabled=enabled,
            source=str(current.get("source") or "user"),
        )
        print(f"{args.intent_name} -> {'enabled' if enabled else 'disabled'}")
        return

    if args.abilities_command == "delete":
        ok = store.delete_spec(args.intent_name)
        if not ok:
            print("Ability spec not found.")
            return
        print(f"Deleted ability spec: {args.intent_name}")
        return


def _load_spec_patch(spec_json: str | None, spec_file: str | None) -> dict[str, object]:
    if spec_json and spec_file:
        raise ValueError("Use either --spec-json or --spec-file, not both.")
    if spec_json:
        try:
            parsed = json.loads(spec_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --spec-json: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--spec-json must be a JSON object.")
        return parsed
    if spec_file:
        path = Path(spec_file)
        if not path.exists():
            raise ValueError(f"Spec file not found: {path}")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Spec file must contain a JSON object: {path}")
        return parsed
    return {}


def _command_onboarding(args: argparse.Namespace) -> None:
    if args.onboarding_command == "list":
        rows = list_onboarding_profiles(state=args.state, limit=args.limit)
        if not rows:
            print("No onboarding profiles found.")
            return
        for row in rows:
            print(
                f"- principal={row.get('principal_id')} state={row.get('state')} "
                f"role={row.get('primary_role')}"
            )
        return

    if args.onboarding_command == "show":
        item = get_onboarding_profile(args.principal_id)
        if not item:
            print("Onboarding profile not found.")
            return
        print(json.dumps(item, indent=2, ensure_ascii=True))
        return

    if args.onboarding_command == "upsert":
        principal_id = upsert_onboarding_profile(
            {
                "principal_id": args.principal_id,
                "state": args.state,
                "primary_role": args.primary_role,
                "next_steps": list(args.next_steps or []),
                "resume_token": args.resume_token,
                "completed_at": args.completed_at,
            }
        )
        print(f"Upserted onboarding profile: {principal_id}")
        return

    if args.onboarding_command == "delete":
        ok = delete_onboarding_profile(args.principal_id)
        if not ok:
            print("Onboarding profile not found.")
            return
        print(f"Deleted onboarding profile: {args.principal_id}")
        return


def _command_locations(args: argparse.Namespace) -> None:
    if args.locations_command == "list":
        rows = list_location_profiles(
            principal_id=args.principal_id,
            label=args.label,
            active_only=args.active_only,
            limit=args.limit,
        )
        if not rows:
            print("No location profiles found.")
            return
        for row in rows:
            print(
                f"- {row.get('location_id')} principal={row.get('principal_id')} "
                f"label={row.get('label')} active={row.get('is_active')}"
            )
        return

    if args.locations_command == "show":
        item = get_location_profile(args.location_id)
        if not item:
            print("Location profile not found.")
            return
        print(json.dumps(item, indent=2, ensure_ascii=True))
        return

    if args.locations_command == "upsert":
        location_id = upsert_location_profile(
            {
                "location_id": args.location_id,
                "principal_id": args.principal_id,
                "label": args.label,
                "address_text": args.address_text,
                "latitude": args.lat,
                "longitude": args.lng,
                "source": args.source,
                "confidence": args.confidence,
                "is_active": not args.inactive,
            }
        )
        print(f"Upserted location profile: {location_id}")
        return

    if args.locations_command == "delete":
        ok = delete_location_profile(args.location_id)
        if not ok:
            print("Location profile not found.")
            return
        print(f"Deleted location profile: {args.location_id}")
        return

    if args.locations_command == "device-add":
        metadata = None
        if args.metadata_json:
            try:
                metadata = json.loads(args.metadata_json)
            except json.JSONDecodeError:
                print("Invalid --metadata-json")
                return
        entry_id = insert_device_location(
            {
                "principal_id": args.principal_id,
                "device_id": args.device_id,
                "latitude": args.lat,
                "longitude": args.lng,
                "accuracy_meters": args.accuracy,
                "source": args.source,
                "observed_at": args.observed_at,
                "metadata": metadata,
            }
        )
        print(f"Inserted device location: {entry_id}")
        return

    if args.locations_command == "device-list":
        rows = list_device_locations(
            principal_id=args.principal_id,
            device_id=args.device_id,
            limit=args.limit,
        )
        if not rows:
            print("No device locations found.")
            return
        for row in rows:
            print(
                f"- {row.get('id')} device={row.get('device_id')} "
                f"lat={row.get('latitude')} lng={row.get('longitude')} observed={row.get('observed_at')}"
            )
        return


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, level_name.upper(), logging.INFO))


if __name__ == "__main__":
    main()
