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
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import apply_seed
from alphonse.agent.cognition.brain_health import BrainUnavailable, require_brain_health
from alphonse.agent.core.settings_store import init_db as init_settings_db
from alphonse.agent.io import get_io_registry
from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.services.pdca_queue_runner import PdcaQueueRunner


def load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def load_heart(config: HeartConfig, bus: Bus, ddfsm: DDFSM) -> Heart:
    return Heart(config, bus=bus, ddfsm=ddfsm)


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

    register_senses(str(db_path))
    register_signals(str(db_path))

    io_registry = get_io_registry()
    logging.info(
        "IO registry ready senses=%s extremities=%s",
        ",".join(sorted(io_registry.senses.keys())),
        ",".join(sorted(io_registry.extremities.keys())),
    )

    sense_manager = SenseManager(db_path=str(db_path), bus=bus)
    pdca_queue_runner = PdcaQueueRunner(bus=bus)
    _emit_pdca_startup_mode(pdca_queue_runner.enabled)
    heart = load_heart(config, bus, ddfsm)
    senses_started = False
    queue_runner_started = False
    try:
        sense_manager.start()
        senses_started = True
        pdca_queue_runner.start()
        queue_runner_started = True
        heart.run()
    except KeyboardInterrupt:
        logging.info("Shutdown requested (KeyboardInterrupt).")
    finally:
        heart.stop()
        bus.emit(Signal(type=SHUTDOWN, source="system"))
        if queue_runner_started:
            pdca_queue_runner.stop()
        if senses_started:
            sense_manager.stop()


def _emit_pdca_startup_mode(slicing_enabled: bool) -> None:
    log = get_log_manager()
    ingress = {
        "telegram": _env_enabled("ALPHONSE_ENABLE_TELEGRAM", default=False),
        "cli": _env_enabled("ALPHONSE_ENABLE_CLI", default=False),
    }
    log.emit(
        event="runtime.pdca_slicing.startup",
        component="main",
        status="enabled" if slicing_enabled else "disabled",
        payload={"ingress": ingress},
    )
    if slicing_enabled:
        return
    if not any(ingress.values()):
        return
    log.emit(
        level="warning",
        event="runtime.pdca_slicing.disabled_with_ingress",
        component="main",
        status="disabled",
        error_code="pdca_slicing_disabled",
        payload={"ingress": ingress},
    )


def _resolve_nerve_db_path() -> Path:
    return resolve_nervous_system_db_path()


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


if __name__ == "__main__":
    main()
