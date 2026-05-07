"""Contracts for integration adapters."""

from .integration_adapter import (
    Action,
    CanonicalInboundEvent,
    CanonicalInboundMessage,
    IntegrationAdapter,
    Signal,
)

__all__ = ["Action", "CanonicalInboundEvent", "CanonicalInboundMessage", "IntegrationAdapter", "Signal"]
