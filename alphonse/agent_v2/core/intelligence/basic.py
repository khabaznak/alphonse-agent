"""Basic placeholder intelligence processor for Alphonse v2."""

from __future__ import annotations

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import ProcessingResult
from alphonse.agent_v2.core.core import StateSnapshot


class BasicIntelligenceProcessor:
    """Small processor used by the first native TUI."""

    def process(self, message: CoreMessage, context: CoreLoopContext) -> ProcessingResult:
        _ = context
        response = _response_for_message(message)
        return ProcessingResult(
            snapshot=StateSnapshot(
                phase=ImprovementPhase.ACT,
                task_owner=message.user,
                current_work=message.prompt,
                metadata={"response": response},
            )
        )


def _response_for_message(message: CoreMessage) -> str:
    is_command = bool(message.metadata.get("is_command"))
    if is_command:
        command = str(message.metadata.get("command") or "")
        args = str(message.metadata.get("command_args") or "")
        suffix = f" {args}" if args else ""
        return f"Command /{command}{suffix} detected. Command execution is not implemented yet."
    return f"I received your message: {message.prompt}"

