"""Model and request contracts for v2 inference routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from alphonse.agent_v2.core.core import ToolDescriptor


class InferencePurpose(str, Enum):
    """Reason a CAPD node is asking for inference."""

    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    CRITERIA_REVIEW = "criteria_review"
    TOOL_PLANNING = "tool_planning"
    TACTICAL_ACTION = "tactical_action"
    PHASE_REVIEW = "phase_review"
    STRATEGIC_ACT = "strategic_act"
    FINAL_RESPONSE = "final_response"
    PHASE_PLANNING = "phase_planning"
    MEMORY_COMPACTION = "memory_compaction"


@dataclass(frozen=True)
class ModelProfile:
    """Resolved model/provider capability profile."""

    provider: str
    model: str
    profile_id: str
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    supports_json_mode: bool = False
    cost_tier: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceRequest:
    """One model request from a CAPD node."""

    prompt: str
    purpose: InferencePurpose
    project_id: str = ""
    user: str | None = None
    task_id: str | None = None
    tools: tuple[ToolDescriptor, ...] = ()
    model_profile: ModelProfile | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_checker: Callable[[], bool] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class InferenceResult:
    """Normalized result returned by an inference provider."""

    content: str = ""
    json_value: Any = None
    tool_call: dict[str, Any] | None = None
    model_profile: ModelProfile | None = None
    raw_response: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
