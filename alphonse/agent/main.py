"""Agent entrypoint."""

from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from alphonse.agent.heart import Heart, HeartConfig, SHUTDOWN
from alphonse.agent.nervous_system.ddfsm import DDFSM, DDFSMConfig
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.senses.manager import SenseManager
from alphonse.agent.nervous_system.senses.registry import (
    register_senses,
    register_signals,
)
from alphonse.infrastructure.api_server import ApiServer
from alphonse.infrastructure.api_gateway import gateway
from alphonse.infrastructure.api_exchange import ApiExchange
from alphonse.agent.relay.client import build_relay_client_from_env
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import apply_seed
from alphonse.agent.cognition.brain_health import BrainUnavailable, require_brain_health
from alphonse.agent.core.settings_store import init_db as init_settings_db
from alphonse.agent.io import get_io_registry
from alphonse.agent.services.pdca_queue_runner import PdcaQueueRunner


def load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def load_heart(config: HeartConfig, bus: Bus, ddfsm: DDFSM) -> Heart:
    return Heart(config, bus=bus, ddfsm=ddfsm, ctx=bus)


def main() -> None:
    load_env()
    log_level = os.getenv("ALPHONSE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    llm_model = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct")
    logging.info("LLM model=%s", llm_model)

    init_settings_db()

    # Resolve once; used across Alphonse components.
    db_path = _resolve_nerve_db_path()
    logging.info("Nerve DB path=%s exists=%s", db_path, db_path.exists())
    # Config / nervous system
    config = HeartConfig(
        nervous_system_db_path=str(db_path),
    )
    ddfsm = DDFSM(DDFSMConfig(db_path=str(db_path)))

    apply_schema(db_path)
    apply_seed(db_path)
    try:
        require_brain_health(db_path)
    except BrainUnavailable as exc:
        logging.critical("Brain health check failed. Shutting down gracefully: %s", exc)
        return

    # Signal bus is in-memory transport only.
    bus = Bus()

    gateway.configure(bus, ApiExchange())

    register_senses(str(db_path))
    register_signals(str(db_path))

    io_registry = get_io_registry()
    logging.info(
        "IO registry ready senses=%s extremities=%s",
        ",".join(sorted(io_registry.senses.keys())),
        ",".join(sorted(io_registry.extremities.keys())),
    )

    sense_manager = SenseManager(db_path=str(db_path), bus=bus)
    sense_manager.start()
    pdca_queue_runner = PdcaQueueRunner(bus=bus)
    pdca_queue_runner.start()

    api_server = _build_api_server()
    if api_server:
        api_server.start()
    relay_client = build_relay_client_from_env()
    if relay_client:
        relay_client.start()
    heart = load_heart(config, bus, ddfsm)
    try:
        heart.run()
    except KeyboardInterrupt:
        logging.info("Shutdown requested (KeyboardInterrupt).")
        heart.stop()
        bus.emit(Signal(type=SHUTDOWN, source="system"))
    finally:
        pdca_queue_runner.stop()
        if api_server:
            api_server.stop()
        if relay_client:
            relay_client.stop()
        sense_manager.stop()


def _build_api_server() -> ApiServer | None:
    enabled = os.getenv("ALPHONSE_ENABLE_API", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None
    host = os.getenv("ALPHONSE_API_HOST", "0.0.0.0")
    port_raw = os.getenv("ALPHONSE_API_PORT", "8001")
    try:
        port = int(port_raw)
    except ValueError:
        port = 8001
    return ApiServer(host=host, port=port)


def _resolve_nerve_db_path() -> Path:
    return resolve_nervous_system_db_path()


if __name__ == "__main__":
    main()
