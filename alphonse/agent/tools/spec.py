from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SafetyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ToolSpec:
    key: str
    description: str
    input_schema: dict[str, Any]
    domain_tags: list[str] = field(default_factory=list)
    safety_level: SafetyLevel = SafetyLevel.LOW
    requires_confirmation: bool = False
    examples: list[dict[str, Any]] = field(default_factory=list)
    when_to_use: str = ""
    returns: str = ""
    expose_in_catalog: bool = True
    expose_in_schemas: bool = True

