"""CLI extremity for interactive input/output."""

from __future__ import annotations

import logging
import os
import threading
import time

from alphonse.agent.cognition.skills.interpretation.interpreter import MessageInterpreter
from alphonse.agent.cognition.skills.interpretation.models import MessageEvent
from alphonse.agent.cognition.skills.interpretation.registry import build_default_registry
from alphonse.agent.cognition.skills.interpretation.skills import SkillExecutor, build_ollama_client

logger = logging.getLogger(__name__)


def build_cli_extremity_from_env() -> "CliExtremity | None":
    enabled = _env_flag("ALPHONSE_ENABLE_CLI")
    if not enabled:
        return None
    return CliExtremity()


class CliExtremity:
    def __init__(self) -> None:
        registry = build_default_registry()
        llm_client = build_ollama_client()
        self._interpreter = MessageInterpreter(registry, llm_client)
        self._executor = SkillExecutor(registry, llm_client)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("CLI extremity started")

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("CLI extremity stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = input("alphonse> ")
            except EOFError:
                self._stop_event.set()
                return
            text = text.strip()
            if not text:
                continue
            event = MessageEvent(
                text=text,
                user_id=None,
                channel="cli",
                timestamp=time.time(),
                metadata={},
            )
            decision = self._interpreter.interpret(event)
            response = self._executor.respond(decision, event)
            print(response)


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
