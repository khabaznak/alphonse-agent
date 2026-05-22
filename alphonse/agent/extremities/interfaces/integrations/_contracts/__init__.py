"""Contracts for integration adapters."""

from .canonical_sense_event import CanonicalSenseEvent
from .integration_adapter import (
    Action,
    CanonicalInboundEvent,
    IntegrationAdapter,
    Signal,
)

__all__ = ["Action", "CanonicalInboundEvent", "CanonicalSenseEvent", "IntegrationAdapter", "Signal"]
