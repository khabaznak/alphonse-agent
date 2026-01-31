from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.api import ApiSense, ApiSignal, build_api_signal
from alphonse.infrastructure.api_exchange import ApiExchange


@dataclass
class ApiGateway:
    bus: Bus | None = None
    exchange: ApiExchange | None = None
    sense: ApiSense | None = None

    def configure(self, bus: Bus, exchange: ApiExchange) -> None:
        self.bus = bus
        self.exchange = exchange
        self.sense = ApiSense()

    def emit_and_wait(self, api_signal: ApiSignal, timeout: float = 5.0) -> dict[str, Any] | None:
        if not self.bus or not self.exchange or not self.sense:
            return None
        self.sense.emit(self.bus, api_signal)
        return self.exchange.wait(api_signal.correlation_id, timeout=timeout)

    def build_signal(self, signal_type: str, payload: dict[str, object] | None, correlation_id: str | None) -> ApiSignal:
        return build_api_signal(signal_type, payload, correlation_id)


gateway = ApiGateway()
