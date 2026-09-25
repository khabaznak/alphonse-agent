"""Versioned state contracts for hierarchical CAPD phases."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

V3_SCHEMA_VERSION = 3
SUPPORTED_COMPLETION_KINDS = ("output_present", "tool_call_terminal", "field_equals")


class PhaseStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUBGOAL_COMPLETE = "subgoal_complete"
    WAITING_USER = "waiting_user"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    PHASE_COMPLETE = "phase_complete"


class FailurePolicy(str, Enum):
    STOP = "stop"
    LOCAL_FALLBACK = "local_fallback"
    WAIT_USER = "wait_user"
    STRATEGIC_REVIEW = "strategic_review"


class SideEffectClass(str, Enum):
    READ_ONLY = "read_only"
    USER_RESPONSE = "user_response"
    PROJECT_MUTATION = "project_mutation"
    EXTERNAL_REVERSIBLE = "external_reversible"
    EXTERNAL_IRREVERSIBLE = "external_irreversible"


_OUTCOME_STATUSES = {
    PhaseStatus.WAITING_USER,
    PhaseStatus.BLOCKED,
    PhaseStatus.CANCELLED,
    PhaseStatus.PHASE_COMPLETE,
}
_IMMUTABLE_STATUSES = _OUTCOME_STATUSES - {PhaseStatus.WAITING_USER}
_TRANSITIONS = {
    PhaseStatus.PLANNED: {PhaseStatus.RUNNING, PhaseStatus.CANCELLED},
    PhaseStatus.RUNNING: {
        PhaseStatus.SUBGOAL_COMPLETE,
        PhaseStatus.WAITING_USER,
        PhaseStatus.BLOCKED,
        PhaseStatus.CANCELLED,
    },
    PhaseStatus.SUBGOAL_COMPLETE: {PhaseStatus.RUNNING, PhaseStatus.PHASE_COMPLETE, PhaseStatus.CANCELLED},
    PhaseStatus.WAITING_USER: {PhaseStatus.RUNNING, PhaseStatus.CANCELLED},
}


@dataclass(frozen=True)
class CompletionCondition:
    kind: str
    output_type: str = ""
    field: str = ""
    expected: Any = None

    def __post_init__(self) -> None:
        if not str(self.kind or "").strip():
            raise ValueError("completion_condition_kind_required")
        if self.kind not in SUPPORTED_COMPLETION_KINDS:
            raise ValueError(f"completion_condition_kind_invalid:{self.kind}")
        if self.kind == "field_equals" and not str(self.field or "").strip():
            raise ValueError("completion_condition_field_required")

    @classmethod
    def from_dict(cls, value: Any) -> "CompletionCondition":
        if not isinstance(value, dict):
            raise ValueError("completion_condition_invalid")
        return cls(
            kind=str(value.get("kind") or "").strip(),
            output_type=str(value.get("output_type") or "").strip(),
            field=str(value.get("field") or "").strip(),
            expected=_json_safe(value.get("expected")),
        )


@dataclass(frozen=True)
class MutationScope:
    allowed_paths: tuple[str, ...] = ()
    allow_external_effects: bool = False

    def __post_init__(self) -> None:
        normalized: list[str] = []
        for raw in self.allowed_paths:
            path = str(raw or "").strip().replace("\\", "/")
            pure = PurePosixPath(path)
            if not path or pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"mutation_scope_path_invalid:{path or '(missing)'}")
            if pure.parts and pure.parts[0] == ".alphonse":
                raise ValueError(f"mutation_scope_path_protected:{path}")
            normalized.append(str(pure))
        if len(set(normalized)) != len(normalized):
            raise ValueError("mutation_scope_duplicate_path")
        object.__setattr__(self, "allowed_paths", tuple(normalized))

    @classmethod
    def from_dict(cls, value: Any) -> "MutationScope":
        raw = value if isinstance(value, dict) else {}
        return cls(
            allowed_paths=_strings(raw.get("allowed_paths")),
            allow_external_effects=bool(raw.get("allow_external_effects", False)),
        )


@dataclass(frozen=True)
class PhaseSubgoal:
    subgoal_id: str
    objective: str
    required_output_type: str
    depends_on: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    allowed_side_effects: tuple[SideEffectClass, ...] = (SideEffectClass.READ_ONLY,)
    completion: CompletionCondition = field(default_factory=lambda: CompletionCondition(kind="output_present"))
    failure_policy: FailurePolicy = FailurePolicy.STOP

    def __post_init__(self) -> None:
        if not self.subgoal_id.strip():
            raise ValueError("phase_subgoal_id_required")
        if not self.objective.strip():
            raise ValueError("phase_subgoal_objective_required")
        if not self.required_output_type.strip():
            raise ValueError("phase_subgoal_output_type_required")
        if self.subgoal_id in self.depends_on:
            raise ValueError(f"phase_subgoal_self_dependency:{self.subgoal_id}")

    @classmethod
    def from_dict(cls, value: Any) -> "PhaseSubgoal":
        if not isinstance(value, dict):
            raise ValueError("phase_subgoal_invalid")
        return cls(
            subgoal_id=str(value.get("subgoal_id") or "").strip(),
            objective=str(value.get("objective") or "").strip(),
            required_output_type=str(value.get("required_output_type") or "").strip(),
            depends_on=_strings(value.get("depends_on")),
            allowed_capabilities=_strings(value.get("allowed_capabilities")),
            allowed_side_effects=tuple(
                SideEffectClass(str(item)) for item in value.get("allowed_side_effects") or [SideEffectClass.READ_ONLY.value]
            ),
            completion=CompletionCondition.from_dict(value.get("completion")),
            failure_policy=FailurePolicy(str(value.get("failure_policy") or FailurePolicy.STOP.value)),
        )


@dataclass(frozen=True)
class PhasePlan:
    phase_id: str
    objective: str
    subgoals: tuple[PhaseSubgoal, ...]
    criterion_ids: tuple[str, ...] = ()
    authorized_capabilities: tuple[str, ...] = ()
    mutation_scope: MutationScope = field(default_factory=MutationScope)
    originating_decision: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = V3_SCHEMA_VERSION
    curated_tool_ids: tuple[str, ...] = ()
    tool_curation_status: str = ""
    acceptance_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != V3_SCHEMA_VERSION:
            raise ValueError(f"phase_schema_version_unsupported:{self.schema_version}")
        if not self.phase_id.strip():
            raise ValueError("phase_id_required")
        if not self.objective.strip():
            raise ValueError("phase_objective_required")
        if not self.subgoals:
            raise ValueError("phase_subgoals_required")
        ids = [item.subgoal_id for item in self.subgoals]
        if len(ids) != len(set(ids)):
            raise ValueError("phase_subgoal_ids_duplicate")
        known: set[str] = set()
        for subgoal in self.subgoals:
            missing = [item for item in subgoal.depends_on if item not in known]
            if missing:
                raise ValueError(f"phase_subgoal_dependency_invalid:{subgoal.subgoal_id}:{','.join(missing)}")
            unknown_capabilities = set(subgoal.allowed_capabilities) - set(self.authorized_capabilities)
            if unknown_capabilities:
                raise ValueError(f"phase_subgoal_capability_unauthorized:{subgoal.subgoal_id}")
            known.add(subgoal.subgoal_id)
        if any(not str(item).strip() for item in self.acceptance_criteria):
            raise ValueError("phase_acceptance_criteria_invalid")

    def to_dict(self) -> dict[str, Any]:
        return _enum_values(asdict(self))

    @classmethod
    def from_dict(cls, value: Any) -> "PhasePlan":
        if not isinstance(value, dict):
            raise ValueError("phase_plan_invalid")
        return cls(
            phase_id=str(value.get("phase_id") or "").strip(),
            objective=str(value.get("objective") or "").strip(),
            subgoals=tuple(PhaseSubgoal.from_dict(item) for item in value.get("subgoals") or []),
            criterion_ids=_strings(value.get("criterion_ids")),
            authorized_capabilities=_strings(value.get("authorized_capabilities")),
            mutation_scope=MutationScope.from_dict(value.get("mutation_scope")),
            originating_decision=str(value.get("originating_decision") or "").strip(),
            created_at=str(value.get("created_at") or "").strip() or datetime.now(timezone.utc).isoformat(),
            schema_version=int(value.get("schema_version", V3_SCHEMA_VERSION)),
            curated_tool_ids=_strings(value.get("curated_tool_ids")),
            tool_curation_status=str(value.get("tool_curation_status") or "").strip(),
            acceptance_criteria=tuple(str(item).strip() for item in value.get("acceptance_criteria") or []),
        )


@dataclass(frozen=True)
class TacticalAction:
    action_id: str
    subgoal_id: str
    tool_id: str
    arguments: dict[str, Any]
    status: str = "planned"
    result: Any = None
    error: str = ""
    acceptance_questions: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id.strip() or not self.subgoal_id.strip() or not self.tool_id.strip():
            raise ValueError("tactical_action_identity_required")
        if self.status not in {"planned", "running", "success", "failed", "waiting"}:
            raise ValueError(f"tactical_action_status_invalid:{self.status}")

    def to_dict(self) -> dict[str, Any]:
        return _enum_values(asdict(self))

    @classmethod
    def from_dict(cls, value: Any) -> "TacticalAction":
        if not isinstance(value, dict):
            raise ValueError("tactical_action_invalid")
        return cls(
            action_id=str(value.get("action_id") or "").strip(),
            subgoal_id=str(value.get("subgoal_id") or "").strip(),
            tool_id=str(value.get("tool_id") or "").strip(),
            arguments=dict(value.get("arguments") or {}),
            status=str(value.get("status") or "planned").strip(),
            result=_json_safe(value.get("result")),
            error=str(value.get("error") or "").strip(),
            acceptance_questions=tuple(
                dict(item) for item in value.get("acceptance_questions") or [] if isinstance(item, dict)
            ),
        )


@dataclass
class PhaseEvidence:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def append(self, entry: dict[str, Any]) -> None:
        payload = _json_safe(entry)
        if not isinstance(payload, dict):
            raise ValueError("phase_evidence_entry_invalid")
        self.entries.append(payload)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [_json_safe(item) for item in self.entries]}

    @classmethod
    def from_dict(cls, value: Any) -> "PhaseEvidence":
        raw = value.get("entries") if isinstance(value, dict) else []
        return cls(entries=[dict(item) for item in raw or [] if isinstance(item, dict)])


@dataclass
class TacticalState:
    phase: PhasePlan
    active_subgoal_id: str
    status: PhaseStatus = PhaseStatus.PLANNED
    completed_subgoal_ids: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    revealed_capabilities: list[str] = field(default_factory=list)
    revealed_tool_ids: list[str] = field(default_factory=list)
    system_one_relevant_tool_ids: list[str] = field(default_factory=list)
    system_one_tool_registry_status: str = ""
    actions: list[TacticalAction] = field(default_factory=list)
    evidence: PhaseEvidence = field(default_factory=PhaseEvidence)
    steering_pending: bool = False
    cancellation_pending: bool = False

    def __post_init__(self) -> None:
        ids = {item.subgoal_id for item in self.phase.subgoals}
        if self.active_subgoal_id not in ids:
            raise ValueError("tactical_active_subgoal_invalid")
        unknown_completed = set(self.completed_subgoal_ids) - ids
        if unknown_completed:
            raise ValueError("tactical_completed_subgoal_invalid")

    def transition(self, status: PhaseStatus) -> None:
        requested = PhaseStatus(status)
        if self.status in _IMMUTABLE_STATUSES:
            raise ValueError(f"phase_status_terminal:{self.status.value}")
        if requested not in _TRANSITIONS.get(self.status, set()):
            raise ValueError(f"phase_transition_invalid:{self.status.value}->{requested.value}")
        self.status = requested

    def bind_subgoal_output(self, subgoal_id: str, output_type: str, value: Any) -> None:
        subgoal = next((item for item in self.phase.subgoals if item.subgoal_id == subgoal_id), None)
        if subgoal is None:
            raise ValueError(f"tactical_binding_subgoal_unknown:{subgoal_id}")
        if output_type != subgoal.required_output_type:
            raise ValueError(f"tactical_binding_output_type_invalid:{output_type}")
        self.bindings[output_type] = {
            "subgoal_id": subgoal_id,
            "output_type": output_type,
            "value": _json_safe(value),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.to_dict(),
            "active_subgoal_id": self.active_subgoal_id,
            "status": self.status.value,
            "completed_subgoal_ids": list(self.completed_subgoal_ids),
            "bindings": _json_safe(self.bindings),
            "revealed_capabilities": list(self.revealed_capabilities),
            "revealed_tool_ids": list(self.revealed_tool_ids),
            "system_one_relevant_tool_ids": list(self.system_one_relevant_tool_ids),
            "system_one_tool_registry_status": self.system_one_tool_registry_status,
            "actions": [item.to_dict() for item in self.actions],
            "evidence": self.evidence.to_dict(),
            "steering_pending": self.steering_pending,
            "cancellation_pending": self.cancellation_pending,
        }

    def prompt_projection(
        self, *, max_evidence_entries: int | None = 8, max_chars: int | None = 12_000
    ) -> str:
        payload = self.to_dict()
        if max_evidence_entries is not None:
            payload["evidence"] = {"entries": self.evidence.entries[-max(1, max_evidence_entries) :]}
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        return rendered if max_chars is None or len(rendered) <= max_chars else rendered[: max(1, max_chars - 18)].rstrip() + '\n"... truncated"'

    @classmethod
    def from_dict(cls, value: Any) -> "TacticalState":
        if not isinstance(value, dict):
            raise ValueError("tactical_state_invalid")
        return cls(
            phase=PhasePlan.from_dict(value.get("phase")),
            active_subgoal_id=str(value.get("active_subgoal_id") or "").strip(),
            status=_phase_status(value.get("status"), default=PhaseStatus.PLANNED),
            completed_subgoal_ids=list(_strings(value.get("completed_subgoal_ids"))),
            bindings=dict(value.get("bindings") or {}),
            revealed_capabilities=list(_strings(value.get("revealed_capabilities"))),
            revealed_tool_ids=list(_strings(value.get("revealed_tool_ids"))),
            system_one_relevant_tool_ids=list(_strings(value.get("system_one_relevant_tool_ids"))),
            system_one_tool_registry_status=str(value.get("system_one_tool_registry_status") or "").strip(),
            actions=[TacticalAction.from_dict(item) for item in value.get("actions") or []],
            evidence=PhaseEvidence.from_dict(value.get("evidence")),
            steering_pending=bool(value.get("steering_pending", False)),
            cancellation_pending=bool(value.get("cancellation_pending", False)),
        )


@dataclass(frozen=True)
class PhaseOutcome:
    phase_id: str
    status: PhaseStatus
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.phase_id.strip():
            raise ValueError("phase_outcome_id_required")
        if self.status not in _OUTCOME_STATUSES:
            raise ValueError("phase_outcome_status_nonterminal")

    def to_dict(self) -> dict[str, Any]:
        return _enum_values(asdict(self))

    @classmethod
    def from_dict(cls, value: Any) -> "PhaseOutcome":
        if not isinstance(value, dict):
            raise ValueError("phase_outcome_invalid")
        return cls(
            phase_id=str(value.get("phase_id") or "").strip(),
            status=_phase_status(value.get("status")),
            reason=str(value.get("reason") or "").strip(),
            evidence_refs=_strings(value.get("evidence_refs")),
            blockers=_strings(value.get("blockers")),
        )


def legacy_call_phase(call: dict[str, Any], *, phase_id: str = "legacy-phase") -> PhasePlan:
    """Wrap one legacy planned call without inferring a richer V3 plan."""
    call_id = str(call.get("id") or "legacy-call").strip()
    tool_id = str(call.get("tool_id") or "legacy_tool").strip()
    return PhasePlan(
        phase_id=phase_id,
        objective=str(call.get("internal_state") or "Execute legacy planned call").strip(),
        subgoals=(
            PhaseSubgoal(
                subgoal_id=call_id,
                objective=str(call.get("internal_state") or "Execute legacy planned call").strip(),
                required_output_type="legacy_tool_result",
                allowed_capabilities=(tool_id,),
                completion=CompletionCondition(kind="tool_call_terminal", output_type="legacy_tool_result"),
            ),
        ),
        authorized_capabilities=(tool_id,),
        originating_decision="legacy_one_call_adapter",
    )


def new_tactical_state(
    phase: PhasePlan,
    *,
    cumulative_evidence: list[dict[str, Any]] | None = None,
) -> TacticalState:
    """Start a phase and import prior evidence without planner-authored budgets."""
    evidence = PhaseEvidence()
    for item in cumulative_evidence or []:
        if isinstance(item, dict):
            evidence.append({"source": "prior_task_evidence", **item})
    return TacticalState(
        phase=phase,
        active_subgoal_id=phase.subgoals[0].subgoal_id,
        evidence=evidence,
    )


def _strings(value: Any) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in value or [] if str(item).strip()) if isinstance(value, (list, tuple)) else ()


def _phase_status(value: Any, *, default: PhaseStatus | None = None) -> PhaseStatus:
    raw = str(value or (default.value if default is not None else ""))
    # V3 checkpoints written before administrator-owned limits were adopted may
    # contain this terminal status. Restore it as an ordinary blocked phase.
    if raw == "budget_exhausted":
        return PhaseStatus.BLOCKED
    return PhaseStatus(raw)


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(value)


def _enum_values(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_values(item) for item in value]
    return value
