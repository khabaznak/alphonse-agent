from __future__ import annotations

from enum import Enum


class IntentCategory(str, Enum):
    CORE_CONVERSATIONAL = "core_conversational"
    TASK_PLANE = "task_plane"
    CONTROL_PLANE = "control_plane"
    DEBUG_META = "debug_meta"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
