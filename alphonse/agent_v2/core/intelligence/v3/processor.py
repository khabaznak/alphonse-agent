"""Hierarchical V3 processor and per-task engine router."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphonse.agent_v2.core.core import ImprovementPhase, ProcessingResult, ProcessingStatus, StateSnapshot
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseStatus, TacticalState, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.outer import V3OuterController
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.intelligence.v3.planner import V3PhasePlanValidationError
from alphonse.agent_v2.system_one import SystemOneUnavailableError
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext, IntelligenceProcessor
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


class HierarchicalCAPDProcessor:
    def __init__(self, *, max_phases: int = 8) -> None:
        self.max_phases = max(1, max_phases)
        self.executor = PhaseExecutor(reveal_policy=ToolRevealPolicy())
        self.outer = V3OuterController()

    def process(self, task: "TaskState", context: "CoreLoopContext") -> ProcessingResult:
        context.emit_ui_event("run_started", {"task": task.to_dict(), "engine": "hierarchical_v3"})
        task.intelligence_engine = "hierarchical_v3"
        task.intelligence_schema_version = 3
        if self._admit_initial_human_task(task, context):
            self._persist(task, context)
            return self._result(task, context)
        phases = 0
        while phases < self.max_phases and task.status not in {"completed", "failed", "waiting_user", "cancelled"}:
            phases += 1
            try:
                if task.hierarchical_state:
                    state = TacticalState.from_dict(task.hierarchical_state)
                    if state.status == PhaseStatus.WAITING_USER:
                        # The question store has already appended the user's answer
                        # to the durable conversation before re-queueing this task.
                        # Re-running the parked subgoal would merely ask the same
                        # question again. Replan from the answer while retaining the
                        # earlier phase evidence in v3_phase_history.
                        task.hierarchical_state = {}
                        state = self._new_state(task, context)
                    elif state.status.value in {"planned", "running"}:
                        pass
                    else:
                        state = self._new_state(task, context)
                else:
                    state = self._new_state(task, context)
                    self._persist(task, context)
            except (V3PhasePlanValidationError, SystemOneUnavailableError) as exc:
                task.status = "failed"
                task.outcome = {"status": "failure", "reason": str(exc)}
                self._persist(task, context)
                break
            try:
                outcome = self.executor.run(task, state, context)
                self._append_history(task, state, outcome.to_dict())
                review, decision = self.outer.review_and_route(task, state, outcome, context)
            except SystemOneUnavailableError as exc:
                task.status = "failed"
                task.outcome = {"status": "failure", "reason": str(exc)}
                self._persist(task, context)
                break
            self._persist(task, context)
            if task.metadata.get("v3_route") in {"plan_next_phase", "strategic_replan"}:
                task.hierarchical_state = {}
                continue
            _ = review, decision
            break
        if phases >= self.max_phases and task.status == "running":
            task.status = "failed"
            task.outcome = {"status": "failure", "reason": "V3 phase budget exhausted without a terminal outcome."}
            self._persist(task, context)
        return self._result(task, context)

    @classmethod
    def _admit_initial_human_task(cls, task: "TaskState", context: "CoreLoopContext") -> bool:
        """Acknowledge a new human task once, or satisfy it with a direct reply."""
        if not _is_initial_human_task(task):
            return False
        existing = task.metadata.get("v3_admission")
        if isinstance(existing, dict) and existing.get("completed") is True:
            return str(existing.get("route") or "") == "direct_response" and task.status == "completed"

        decision_metadata: dict[str, object]
        decision = None
        classify = getattr(context.system_one, "classify_task_admission", None) if context.system_one is not None else None
        if callable(classify):
            try:
                decision = classify(message=task.goal)
            except Exception as exc:
                decision_metadata = {"status": "fallback_to_task", "error_type": type(exc).__name__}
            else:
                decision_metadata = {
                    "status": "used" if bool(getattr(decision, "confident", False)) else "ambiguous_fallback",
                    **decision.to_metadata(),
                }
        else:
            decision_metadata = {"status": "fallback_to_task", "error_type": "classifier_unavailable"}

        direct_response = bool(
            decision is not None
            and bool(getattr(decision, "confident", False))
            and not bool(getattr(decision, "requires_task", True))
        )
        if direct_response:
            try:
                cls._complete_direct_response(task, context)
            except Exception as exc:
                if context.is_cancelled():
                    raise
                decision_metadata["direct_response_error_type"] = type(exc).__name__
                decision_metadata["status"] = "direct_response_failed_fallback_to_task"
            else:
                task.metadata["v3_admission"] = {
                    "completed": True,
                    "route": "direct_response",
                    **decision_metadata,
                }
                context.emit_telemetry({
                    "event": "v3_task_admission", "task_id": task.task_id,
                    **task.metadata["v3_admission"],
                })
                return True

        acknowledgement = cls._deliver_early_acknowledgement(task, context)
        task.metadata["v3_admission"] = {
            "completed": True,
            "route": "task",
            **decision_metadata,
            "acknowledgement": acknowledgement,
        }
        context.emit_activity(
            phase=ImprovementPhase.PLAN,
            label="request acknowledged",
            message=str(acknowledgement.get("message") or "Request received."),
            progress={"engine": "hierarchical_v3", "route": "task", "admission": "completed"},
        )
        context.emit_telemetry({
            "event": "v3_task_admission", "task_id": task.task_id,
            **task.metadata["v3_admission"],
        })
        cls._persist(task, context)
        return False

    @staticmethod
    def _complete_direct_response(task: "TaskState", context: "CoreLoopContext") -> None:
        if context.inference is None:
            raise RuntimeError("v3_direct_response_inference_unavailable")
        context.emit_activity(
            phase=ImprovementPhase.ACT,
            label="responding",
            message="Preparing a direct conversational response.",
            progress={"engine": "hierarchical_v3", "route": "direct_response"},
        )
        result = context.inference.generate_markdown(InferenceRequest(
            prompt=(
                "Reply directly to this conversational message. Be warm, brief, natural, and use the user's language. "
                "The reply must fully satisfy the message without mentioning planning, tools, acceptance criteria, "
                "or internal processing. Do not claim that any external action occurred.\n\n"
                f"User message: {task.goal}"
            ),
            purpose=InferencePurpose.FINAL_RESPONSE,
            project_id=task.project_id,
            user=task.user,
            task_id=task.task_id,
            tools=(),
            cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
        ))
        message = str(result.content or "").strip()
        if not message:
            raise ValueError("v3_direct_response_empty")
        task.metadata["prepared_user_response"] = {"source": "v3_direct_response", "message": message}
        task.metadata["v3_route"] = "respond_and_end"
        task.status = "completed"
        task.outcome = {
            "status": "success",
            "reason": "The message was fully satisfied by one direct conversational response.",
        }

    @staticmethod
    def _deliver_early_acknowledgement(task: "TaskState", context: "CoreLoopContext") -> dict[str, object]:
        message = _acknowledgement_text(task)
        if context.delivery_sink is None:
            return {"status": "delivery_unavailable", "message": message}
        try:
            result = context.delivery_sink({
                "event_type": "task.acknowledge",
                "task": task.to_dict(),
                "message": message,
                "idempotency_key": f"v3-early-ack:{task.task_id or task.message_id or ''}",
            })
        except Exception as exc:
            return {"status": "delivery_failed", "error_type": type(exc).__name__, "message": message}
        rendered = dict(result) if isinstance(result, dict) else {"status": "delivery_unknown"}
        rendered["message"] = message
        return rendered

    @staticmethod
    def _new_state(task: "TaskState", context: "CoreLoopContext") -> TacticalState:
        phase = plan_phase(task, context)
        prior = _deduplicated_history_evidence(task.metadata.get("v3_phase_history"))
        return new_tactical_state(phase, cumulative_evidence=prior)

    @staticmethod
    def _append_history(task: "TaskState", state: TacticalState, outcome: dict) -> None:
        history = task.metadata.setdefault("v3_phase_history", [])
        if isinstance(history, list):
            local_evidence = [
                dict(entry) for entry in state.evidence.entries
                if str(entry.get("phase_id") or "") == state.phase.phase_id
            ]
            history.append({
                "phase": state.phase.to_dict(),
                "outcome": outcome,
                "evidence": {"entries": local_evidence},
            })

    @staticmethod
    def _persist(task: "TaskState", context: "CoreLoopContext") -> None:
        if context.question_store is not None:
            checkpoint_status = "done" if task.status == "completed" else task.status
            context.question_store.save_task_checkpoint(task, status=checkpoint_status)

    @staticmethod
    def _result(task: "TaskState", context: "CoreLoopContext") -> ProcessingResult:
        status = (
            ProcessingStatus.PARKED if task.status == "waiting_user" else
            ProcessingStatus.FAILED if task.status == "failed" else
            ProcessingStatus.CANCELLED if task.status == "cancelled" else ProcessingStatus.COMPLETED
        )
        context.emit_ui_event("run_finished", {"task": task.to_dict(), "status": task.status, "engine": "hierarchical_v3"})
        return ProcessingResult(
            snapshot=StateSnapshot(
                phase=ImprovementPhase.ACT,
                task_owner=task.user,
                current_work=task.goal,
                metadata={
                    "engine": "hierarchical_v3",
                    "status": task.status,
                    "outcome": task.outcome,
                    "phase_progress": task.hierarchical_state,
                    "task_state": task.to_dict(),
                },
            ),
            status=status,
            error=(
                (
                    str((task.outcome or {}).get("reason"))
                    if str((task.outcome or {}).get("reason") or "").startswith("system_one_unavailable")
                    else f"v3_task_failed:{str((task.outcome or {}).get('reason') or 'deterministic_failure')}"
                )
                if status == ProcessingStatus.FAILED else None
            ),
        )


class EngineRoutingProcessor:
    """Delegate to the engine stamped on the queued task."""

    def __init__(self, *, v2: "IntelligenceProcessor", v3: "IntelligenceProcessor") -> None:
        self.v2 = v2
        self.v3 = v3

    def process(self, task: "TaskState", context: "CoreLoopContext") -> ProcessingResult:
        return (self.v3 if task.intelligence_engine == "hierarchical_v3" else self.v2).process(task, context)


def _is_initial_human_task(task: "TaskState") -> bool:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if str(metadata.get("routing_disposition") or "") != "pdca_task":
        return False
    if str(metadata.get("source") or "") in {"scheduled_task", "event_automation"}:
        return False
    task_id = str(task.task_id or "").strip()
    message_id = str(task.message_id or "").strip()
    return bool(task_id and message_id and task_id == message_id)


def _acknowledgement_text(task: "TaskState") -> str:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    channel = metadata.get("channel") if isinstance(metadata.get("channel"), dict) else {}
    locale = str(metadata.get("locale") or metadata.get("language") or channel.get("locale") or "").lower()
    if locale == "es" or locale.startswith("es-") or locale.startswith("es_"):
        return "Entendido — voy a revisarlo con atención."
    return "Got it — I’m taking a closer look now."


def _deduplicated_history_evidence(raw_history: object) -> list[dict]:
    evidence_by_ref: dict[str, dict] = {}
    unreferenced: list[dict] = []
    for item in raw_history if isinstance(raw_history, list) else []:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence")
        entries = evidence.get("entries") if isinstance(evidence, dict) else []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            evidence_ref = str(entry.get("evidence_ref") or "").strip()
            if evidence_ref:
                evidence_by_ref[evidence_ref] = dict(entry)
            else:
                unreferenced.append(dict(entry))
    return [*evidence_by_ref.values(), *unreferenced]
