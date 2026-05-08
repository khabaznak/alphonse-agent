"""Contracts for integration adapters."""

from .integration_adapter import (
    Action,
    CanonicalInboundEvent,
    IntegrationAdapter,
    Signal,
)

__all__ = ["Action", "CanonicalInboundEvent", "IntegrationAdapter", "Signal"]
