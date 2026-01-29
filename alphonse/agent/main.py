"""Agent entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from alphonse.agent.heart import Heart, HeartConfig
from alphonse.nervous_system.ddfsm import DDFSM, DDFSMConfig
from alphonse.senses.bus import Bus
from alphonse.senses.manager import SenseManager
from alphonse.senses.registry import register_senses, register_signals


def load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def resolve_nervous_system_db_path() -> Path:
    default_path = Path(__file__).resolve().parent.parent / "nervous_system" / "db" / "nerve-db"
    configured = os.getenv("NERVE_DB_PATH")
    if not configured:
        return default_path
    configured_path = Path(configured)
    if configured_path.is_absolute():
        return configured_path
    return (Path(__file__).resolve().parent / configured_path).resolve()


def load_heart(config: HeartConfig, bus: Bus, ddfsm: DDFSM) -> Heart:
    return Heart(config, bus=bus, ddfsm=ddfsm)


def main() -> None:
    load_env()

    # Resolve once; used across Alphonse components.
    db_path = resolve_nervous_system_db_path()
    tick_raw = os.getenv("HEART_TICK_SECONDS", "5")
    try:
        tick_seconds = float(tick_raw)
    except ValueError:
        tick_seconds = 5.0

    # Config / nervous system
    config = HeartConfig(nervous_system_db_path=str(db_path), tick_seconds=tick_seconds)
    ddfsm = DDFSM(DDFSMConfig(db_path=str(db_path)))

    # Signal bus is in-memory transport only.
    bus = Bus()

    register_senses(str(db_path))
    register_signals(str(db_path))

    sense_manager = SenseManager(db_path=str(db_path), bus=bus)
    sense_manager.start()

    heart = load_heart(config, bus, ddfsm)
    try:
        heart.run()
    finally:
        sense_manager.stop()


if __name__ == "__main__":
    main()
