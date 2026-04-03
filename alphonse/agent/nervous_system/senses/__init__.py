"""Senses package."""

from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.user_communication import (
    CanonicalUserCommunication,
    UserCommunicationSense,
    build_canonical_user_message,
)

__all__ = [
    "Sense",
    "SignalSpec",
    "CanonicalUserCommunication",
    "UserCommunicationSense",
    "build_canonical_user_message",
]
