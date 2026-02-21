from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable

from pydantic import ValidationError

from alphonse.agent.cognition.narration.outbound_narration_orchestrator import (
    DeliveryCoordinator,
    build_default_coordinator,
)
from alphonse.agent.cognition.plan_execution.communication_dispatcher import CommunicationDispatcher
from alphonse.agent.cognition.plans import CortexPlan
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry
from alphonse.agent.tools.base import ensure_tool_result

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


class ToolMap:
    def __init__(self, *, registry: ToolRegistry) -> None:
        self._registry = registry

    def call(
        self,
        tool_key: str,
        params: dict[str, Any],
        execution_exception_handler: Callable[[Exception], None],
        tool_call_output: Callable[[Any], None],
    ) -> None:
        try:
            tool = self._registry.get(tool_key)
            if tool is None:
                raise RuntimeError(f"tool_not_found:{tool_key}")
            raw = tool.execute(**dict(params or {}))
            result = ensure_tool_result(tool_key=tool_key, value=raw)
            tool_call_output(result)
        except Exception as exc:
            execution_exception_handler(exc)


class PlanExecutor:
    def __init__(
        self,
        *,
        coordinator: DeliveryCoordinator | None = None,
        extremities: object | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        _ = extremities
        self._coordinator = coordinator or build_default_coordinator()
        self._dispatcher = CommunicationDispatcher(coordinator=self._coordinator, logger=logger)
        self._tool_map = ToolMap(registry=tool_registry or build_default_tool_registry())

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
        tool_key = _tool_key(plan)
        params = _plan_params(plan)

        def _on_exception(exc: Exception) -> None:
            raise exc

        def _on_output(result: Any) -> None:
            self._handle_tool_output(
                plan=plan,
                tool_key=tool_key,
                params=params,
                result=result,
                context=context,
                exec_context=exec_context,
            )

        self._tool_map.call(
            tool_key=tool_key,
            params=params,
            execution_exception_handler=_on_exception,
            tool_call_output=_on_output,
        )

    def _handle_tool_output(
        self,
        *,
        plan: CortexPlan,
        tool_key: str,
        params: dict[str, Any],
        result: Any,
        context: dict[str, Any],
        exec_context: PlanExecutionContext,
    ) -> None:
        if tool_key == "communicate":
            message = str(params.get("message") or "").strip()
            if not message:
                raise ValueError("message_required")
            channels = plan.channels or [exec_context.channel_type]
            target = plan.target or exec_context.channel_target
            for channel in channels:
                self._dispatcher.dispatch_step_message(
                    channel=channel,
                    target=target,
                    message=message,
                    context=context,
                    exec_context=exec_context,
                    plan=plan,
                )
            return
        logger.info(
            "executor dispatch plan_id=%s step=%s outcome=tool_called",
            plan.plan_id,
            tool_key,
        )
        _ = result

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


def _tool_key(plan: CortexPlan) -> str:
    return str(getattr(plan, "tool", "") or "").strip().lower()


def _plan_params(plan: CortexPlan) -> dict[str, Any]:
    params = getattr(plan, "parameters", None)
    if isinstance(params, dict):
        return dict(params)
    payload = getattr(plan, "payload", None)
    if isinstance(payload, dict):
        return dict(payload)
    return {}
