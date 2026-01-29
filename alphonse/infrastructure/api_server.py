from __future__ import annotations

import threading
from dataclasses import dataclass

import uvicorn

from alphonse.infrastructure.api import app


@dataclass
class ApiServer:
    host: str
    port: int

    def __post_init__(self) -> None:
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
