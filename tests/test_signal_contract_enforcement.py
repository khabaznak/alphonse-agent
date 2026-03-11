from __future__ import annotations

import pytest

from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.bus import Signal


def test_bus_rejects_unknown_signal_type_for_sense_source() -> None:
    bus = Bus()
    with pytest.raises(ValueError, match="invalid_signal_type_for_source"):
        bus.emit(
            Signal(
                type="telegram.message_received",
                payload={},
                source="telegram",
                correlation_id="c-1",
            )
        )


def test_bus_accepts_declared_signal_type_for_sense_source() -> None:
    bus = Bus()
    bus.emit(
        Signal(
            type="sense.telegram.message.user.received",
            payload={"schema_version": "1.0"},
            source="telegram",
            correlation_id="c-1",
        )
    )


def test_bus_accepts_timed_conscious_payload_for_timer_source() -> None:
    bus = Bus()
    bus.emit(
        Signal(
            type="timed_signal.conscious_payload",
            payload={"schema_version": "1.0"},
            source="timer",
            correlation_id="c-2",
        )
    )
