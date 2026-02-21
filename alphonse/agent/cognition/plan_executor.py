from __future__ import annotations

from dataclasses import dataclass
import logging
from pydantic import ValidationError

from alphonse.agent.cognition.narration.outbound_narration_orchestrator import (
    DeliveryCoordinator,
    build_default_coordinator,
)
from alphonse.agent.cognition.plan_execution.communication_dispatcher import CommunicationDispatcher
from alphonse.agent.cognition.plan_execution import handlers as plan_handlers
from alphonse.agent.cognition.plans import (
    CortexPlan,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanExecutionContext:
    channel_type: str
    channel_target: str | None
    actor_person_id: str | None
    correlation_id: str


@dataclass(frozen=True)
class PlanDispatchFailure:
    plan_id: str
    step: str
    code: str
    message: str
    retryable: bool


class PlanExecutor:
    def __init__(
        self,
        *,
        coordinator: DeliveryCoordinator | None = None,
        extremities: object | None = None,
    ) -> None:
        _ = extremities
        self._coordinator = coordinator or build_default_coordinator()
        self._dispatcher = CommunicationDispatcher(coordinator=self._coordinator, logger=logger)

    def execute(
        self, plans: list[CortexPlan], context: dict, exec_context: PlanExecutionContext
    ) -> None:
        for plan in plans:
            try:
                self._execute_plan(plan, context, exec_context)
            except Exception as exc:
                failure = _normalize_dispatch_exception(plan=plan, exc=exc)
                logger.exception(
                    "executor dispatch failed plan_id=%s step=%s code=%s retryable=%s message=%s",
                    failure.plan_id,
                    failure.step,
                    failure.code,
                    failure.retryable,
                    failure.message,
                )
                self._dispatch_error(exec_context, context, failure=failure)

    def _execute_plan(
        self, plan: CortexPlan, context: dict, exec_context: PlanExecutionContext
    ) -> None:
        handled = plan_handlers.execute_plan(
            logger=logger,
            plan=plan,
            context=context,
            exec_context=exec_context,
            dispatch_message=self._dispatcher.dispatch_step_message,
        )
        if handled:
            return
        logger.info(
            "executor dispatch plan_id=%s step=%s outcome=noop",
            plan.plan_id,
            str(plan.tool or "unknown"),
        )

    def _dispatch_error(
        self,
        exec_context: PlanExecutionContext,
        context: dict,
        *,
        failure: PlanDispatchFailure | None = None,
    ) -> None:
        payload = None
        if failure is not None:
            payload = {
                "code": failure.code,
                "message": failure.message,
                "retryable": failure.retryable,
                "plan_id": failure.plan_id,
                "step": failure.step,
            }
        self._dispatcher.dispatch_execution_error(
            exec_context=exec_context,
            context=context,
            failure=payload,
        )


def _normalize_dispatch_exception(*, plan: CortexPlan, exc: Exception) -> PlanDispatchFailure:
    plan_id = str(plan.plan_id)
    step = str(plan.tool or "unknown")
    if isinstance(exc, ValidationError):
        return PlanDispatchFailure(
            plan_id=plan_id,
            step=step,
            code="invalid_plan_payload",
            message=str(exc),
            retryable=False,
        )
    return PlanDispatchFailure(
        plan_id=plan_id,
        step=step,
        code="executor_dispatch_exception",
        message=str(exc) or type(exc).__name__,
        retryable=True,
    )
