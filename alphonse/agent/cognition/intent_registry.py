from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class IntentCategory(str, Enum):
    CORE_CONVERSATIONAL = "core_conversational"
    TASK_PLANE = "task_plane"
    CONTROL_PLANE = "control_plane"
    DEBUG_META = "debug_meta"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class IntentMetadata:
    category: IntentCategory
    requires_planner: bool
    default_risk: RiskLevel
    supports_autonomy: bool
    patterns: tuple[str, ...] = field(default_factory=tuple)


class IntentRegistry:
    def __init__(self) -> None:
        self._intents: dict[str, IntentMetadata] = {}

    def register(
        self,
        intent_name: str,
        metadata: IntentMetadata,
        *,
        allow_core: bool = False,
    ) -> None:
        if metadata.category == IntentCategory.CORE_CONVERSATIONAL and not allow_core:
            raise ValueError("CORE_CONVERSATIONAL intents require allow_core=True")
        self._intents[intent_name] = metadata

    def get(self, intent_name: str) -> IntentMetadata | None:
        return self._intents.get(intent_name)

    def list_intents(self) -> list[str]:
        return sorted(self._intents.keys())

    def by_category(self, category: IntentCategory) -> dict[str, IntentMetadata]:
        return {
            name: meta
            for name, meta in self._intents.items()
            if meta.category == category
        }

    def all(self) -> dict[str, IntentMetadata]:
        return dict(self._intents)


_GLOBAL_REGISTRY = IntentRegistry()


def get_registry() -> IntentRegistry:
    return _GLOBAL_REGISTRY


def register_builtin_intents(registry: IntentRegistry) -> None:
    registry.register(
        "greeting",
        IntentMetadata(
            category=IntentCategory.CORE_CONVERSATIONAL,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=False,
            patterns=(
                r"\b(hi|hello|hey|good morning|good afternoon|good evening|hola|buenos dias|buenos días|buenas)\b",
            ),
        ),
        allow_core=True,
    )
    registry.register(
        "help",
        IntentMetadata(
            category=IntentCategory.CORE_CONVERSATIONAL,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=False,
            patterns=(r"\b(ayuda|help)\b",),
        ),
        allow_core=True,
    )
    registry.register(
        "identity_question",
        IntentMetadata(
            category=IntentCategory.CORE_CONVERSATIONAL,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=False,
            patterns=(
                r"\b(quien eres|quién eres|who are you|what are you|what is your name|what's your name|como te llamas|cómo te llamas)\b",
            ),
        ),
        allow_core=True,
    )
    registry.register(
        "user_identity_question",
        IntentMetadata(
            category=IntentCategory.CORE_CONVERSATIONAL,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=False,
            patterns=(
                r"\b(quien soy yo|quién soy yo|quien soy|quién soy|cual es mi nombre|cuál es mi nombre|como me llamo|cómo me llamo|who am i|what is my name|what's my name)\b",
            ),
        ),
        allow_core=True,
    )
    registry.register(
        "identity.query_user_name",
        IntentMetadata(
            category=IntentCategory.CORE_CONVERSATIONAL,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=False,
            patterns=(
                r"\b(ya\s+sabes\s+mi\s+nombre|sabes\s+como\s+me\s+llamo|sabes\s+mi\s+nombre|conoces\s+mi\s+nombre)\b",
                r"\b(do you know my name|do you know what my name is|do you know who i am)\b",
            ),
        ),
        allow_core=True,
    )
    registry.register(
        "update_preferences",
        IntentMetadata(
            category=IntentCategory.CONTROL_PLANE,
            requires_planner=True,
            default_risk=RiskLevel.MEDIUM,
            supports_autonomy=True,
        ),
    )
    registry.register(
        "lan.arm",
        IntentMetadata(
            category=IntentCategory.CONTROL_PLANE,
            requires_planner=False,
            default_risk=RiskLevel.HIGH,
            supports_autonomy=False,
            patterns=(r"\b(arm|arm link|unlock|arm alphonse link)\b",),
        ),
    )
    registry.register(
        "lan.disarm",
        IntentMetadata(
            category=IntentCategory.CONTROL_PLANE,
            requires_planner=False,
            default_risk=RiskLevel.HIGH,
            supports_autonomy=False,
            patterns=(r"\b(disarm|lock|disarm link)\b",),
        ),
    )
    registry.register(
        "pair.approve",
        IntentMetadata(
            category=IntentCategory.CONTROL_PLANE,
            requires_planner=False,
            default_risk=RiskLevel.MEDIUM,
            supports_autonomy=False,
            patterns=(r"^(?:/approve|approve)\b",),
        ),
    )
    registry.register(
        "pair.deny",
        IntentMetadata(
            category=IntentCategory.CONTROL_PLANE,
            requires_planner=False,
            default_risk=RiskLevel.MEDIUM,
            supports_autonomy=False,
            patterns=(r"^(?:/deny|deny)\b",),
        ),
    )
    registry.register(
        "get_status",
        IntentMetadata(
            category=IntentCategory.DEBUG_META,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=True,
            patterns=(r"\b(status|estado)\b",),
        ),
    )
    registry.register(
        "meta.capabilities",
        IntentMetadata(
            category=IntentCategory.DEBUG_META,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=True,
            patterns=(
                r"\b(what else can you do|what can you do|capabilities|que puedes hacer|qué puedes hacer|que mas puedes hacer|qué más puedes hacer|que sabes hacer|qué sabes hacer|que mas sabes hacer|qué más sabes hacer)\b",
            ),
        ),
    )
    registry.register(
        "meta.gaps_list",
        IntentMetadata(
            category=IntentCategory.DEBUG_META,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=True,
            patterns=(r"\b(gaps\??|gap list|gaps list|lista de brechas|brechas)\b",),
        ),
    )
    registry.register(
        "timed_signals.list",
        IntentMetadata(
            category=IntentCategory.TASK_PLANE,
            requires_planner=False,
            default_risk=RiskLevel.LOW,
            supports_autonomy=True,
            patterns=(
                r"\b(what reminders do you have|reminders scheduled|list reminders|que recordatorios tienes|qué recordatorios tienes|recordatorios programados)\b",
            ),
        ),
    )
    registry.register(
        "schedule_reminder",
        IntentMetadata(
            category=IntentCategory.TASK_PLANE,
            requires_planner=True,
            default_risk=RiskLevel.LOW,
            supports_autonomy=True,
        ),
    )
    registry.register(
        "unknown",
        IntentMetadata(
            category=IntentCategory.TASK_PLANE,
            requires_planner=True,
            default_risk=RiskLevel.MEDIUM,
            supports_autonomy=False,
        ),
    )


register_builtin_intents(_GLOBAL_REGISTRY)
