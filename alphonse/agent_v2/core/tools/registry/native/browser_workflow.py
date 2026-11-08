"""Disabled-by-default native boundary for the browser workflow foundation."""
from __future__ import annotations

from typing import Any
from copy import deepcopy

from alphonse.agent_v2.core.core import CoreUiEvent, ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition


def build_browser_workflow_tools(*, enabled: bool = False) -> tuple[ToolDefinition, ...]:
    definitions = []
    for operation in ("begin", "reacquire", "observe", "observe_result", "request_value", "fill", "submit", "reconcile", "verify_submission"):
        properties = {"target": {"type": "string", "pattern": "^[0-9]+_[0-9]+$"}} if operation in {"fill", "submit"} else {}
        if operation in {"fill", "request_value"}:
            properties["field"] = {"type": "string", "enum": ["identifier", "vehicle_plate", "vehicle_serial", "name", "number", "other"]}
        schema = {"type": "object", "additionalProperties": False, "properties": properties,
                  "required": (["field", "target"] if operation == "fill" else ["target"] if operation == "submit" else ["field"] if operation == "request_value" else [])}
        descriptor = ToolDescriptor(tool_id=f"native.browser_workflow_{operation}",
            name=f"browser_workflow_{operation}", kind=ToolKind.NATIVE,
            description="Use the task-scoped browser workflow foundation.", argument_schema=schema,
            capabilities=("browser",), tags=("native", "browser"),
            metadata={"sensitive_content": True, "v3_capabilities": ["browser"],
                      "side_effect_class": "external_irreversible" if operation in {"fill", "submit"} else "user_response" if operation == "request_value" else "read_only"},
            read_only=operation not in {"fill", "submit", "request_value"})
        def invoke(arguments: dict[str, Any], *, context: ToolExecutionContext, op: str = operation) -> dict[str, str]:
            if context is None or context.browser_workflow_service is None:
                raise PermissionError("browser_workflow_unavailable")
            service = context.browser_workflow_service
            if op == "begin":
                service.begin(context.task)
            elif op == "reacquire":
                service.reacquire(context.task)
            elif op == "observe":
                service.observe(context.task)
            elif op == "observe_result":
                service.observe(context.task, result=True)
            elif op == "request_value":
                return _request_value(context, service, arguments["field"])
            elif op == "fill":
                service.fill(context.task, field=arguments["field"], target=arguments["target"])
            elif op == "submit":
                service.submit(context.task, target=arguments["target"])
            elif op == "reconcile":
                service.reconcile_fill(context.task)
            else:
                service.verify_submission(context.task)
            return {"status": "performed_unverified"}
        definitions.append(ToolDefinition(descriptor=descriptor, callable=invoke, argument_schema=schema,
                                          enabled=enabled, accepts_context=True))
    return tuple(definitions)


def _request_value(context: ToolExecutionContext, service: Any, field: str) -> dict[str, Any]:
    task = context.task
    if context.question_store is None:
        raise ValueError("browser_private_question_unavailable")
    service.validate_private_request(task, field=field)
    labels = {"identifier": "identifier", "vehicle_plate": "vehicle plate number", "vehicle_serial": "vehicle serial number",
              "name": "name", "number": "number", "other": "value"}
    question_text = f"Enter the {labels[field]} needed for this browser task using the private input channel."
    original = deepcopy(task.__dict__)
    try:
        if context.cancellation_checker and context.cancellation_checker():
            raise PermissionError("browser_private_question_cancelled")
        question = context.question_store.create_question(task=task, question=question_text, kind="open_text",
            respondent_user_id=task.user, expires_in_seconds=900,
            delivery_metadata={"private_browser_value": True, "browser_private_field": field,
                               "origin_channel": dict(task.metadata.get("channel") or {})})
    except BaseException:
        task.__dict__.clear()
        task.__dict__.update(original)
        raise
    if context.cancellation_checker and context.cancellation_checker():
        context.question_store.cancel_question(question.question_id)
        raise PermissionError("browser_private_question_cancelled")
    delivery = None
    if context.delivery_sink is not None:
        try:
            delivery = context.delivery_sink({"event_type": "question.deliver", "question": question.to_dict(), "task": task.to_dict()})
            if isinstance(delivery, dict) and delivery.get("exception"):
                raise RuntimeError("browser_private_question_delivery_failed")
            if isinstance(delivery, dict) and delivery:
                context.question_store.bind_delivery_metadata(question_id=question.question_id, metadata=delivery)
        except Exception:
            context.question_store.cancel_question(question.question_id)
            raise RuntimeError("browser_private_question_delivery_failed") from None
    if context.ui_event_sink is not None:
        context.ui_event_sink(CoreUiEvent(event_type="question_interrupt_opened",
            payload={"question": question.to_dict(), "delivery_result": delivery if isinstance(delivery, dict) else None}))
    return {"waiting_for_answer": True}
