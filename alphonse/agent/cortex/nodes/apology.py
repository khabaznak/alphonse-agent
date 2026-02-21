from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable

from alphonse.agent.cognition.prompt_templates_runtime import (
    CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
    CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
    render_prompt_template,
)
from alphonse.agent.cognition.preferences.store import get_or_create_principal_for_channel
import alphonse.agent.cortex.nodes.execution_helpers as _execution_helpers
from alphonse.agent.cortex.nodes.capability_gap import run_capability_gap_tool as _run_capability_gap_tool_core
from alphonse.agent.cortex.nodes.telemetry import emit_brain_state
from alphonse.agent.cortex.transitions import emit_transition_event as _emit_transition_event

logger = logging.getLogger(__name__)


def apology_node(
    state: dict[str, Any],
    *,
    build_capability_gap_apology: Callable[..., str],
    llm_client: Any,
) -> dict[str, Any]:
    emit_brain_state(
        state=state,
        node="apology_node",
        updates={},
        stage="start",
    )

    def _return(payload: dict[str, Any]) -> dict[str, Any]:
        return emit_brain_state(
            state=state,
            node="apology_node",
            updates=payload,
        )

    if state.get("response_text"):
        return _return({})
    plans = state.get("plans")
    if not isinstance(plans, list):
        return _return({})
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        if str(plan.get("tool") or "").strip().lower() != "capability_gap":
            continue
        payload = (
            plan.get("parameters") if isinstance(plan.get("parameters"), dict) else {}
        ) or (plan.get("payload") if isinstance(plan.get("payload"), dict) else {})
        reason = str(payload.get("reason") or "capability_gap")
        missing_slots = payload.get("missing_slots")
        apology = build_capability_gap_apology(
            state=state,
            llm_client=llm_client,
            reason=reason,
            missing_slots=missing_slots if isinstance(missing_slots, list) else None,
        )
        if apology:
            return _return({"response_text": apology})
        return _return({})
    return _return({})


def apology_node_stateful(
    state: dict[str, Any],
    *,
    build_capability_gap_apology: Callable[..., str],
    llm_client_from_state: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    return apology_node(
        state,
        build_capability_gap_apology=build_capability_gap_apology,
        llm_client=llm_client_from_state(state),
    )


def build_apology_node(
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return partial(
        apology_node_stateful,
        build_capability_gap_apology=lambda **kwargs: _execution_helpers.build_capability_gap_apology(
            **kwargs,
            render_prompt_template=render_prompt_template,
            apology_user_template=CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
            apology_system_prompt=CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
            locale_for_state=lambda s: (
                s.get("locale")
                if isinstance(s.get("locale"), str) and s.get("locale")
                else "en-US"
            ),
            logger_exception=lambda msg, chat_id, correlation_id, rsn: logger.exception(
                msg,
                chat_id,
                correlation_id,
                rsn,
            ),
        ),
        llm_client_from_state=llm_client_from_state,
    )


def run_capability_gap_tool(
    state: dict[str, Any],
    *,
    llm_client: Any,
    reason: str,
    missing_slots: list[str] | None = None,
    append_existing_plans: bool = False,
) -> dict[str, Any]:
    return _run_capability_gap_tool_core(
        state=state,
        llm_client=llm_client,
        reason=reason,
        missing_slots=missing_slots,
        append_existing_plans=append_existing_plans,
        emit_transition_event=_emit_transition_event,
        logger_info=lambda msg, chat_id, correlation_id, rsn: logger.info(
            msg,
            chat_id,
            correlation_id,
            rsn,
        ),
        build_capability_gap_apology=lambda **kwargs: _execution_helpers.build_capability_gap_apology(
            **kwargs,
            render_prompt_template=render_prompt_template,
            apology_user_template=CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
            apology_system_prompt=CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
            locale_for_state=lambda s: (
                s.get("locale") if isinstance(s.get("locale"), str) and s.get("locale") else "en-US"
            ),
            logger_exception=lambda msg, chat_id, correlation_id, rsn: logger.exception(
                msg,
                chat_id,
                correlation_id,
                rsn,
            ),
        ),
        get_or_create_principal_for_channel=get_or_create_principal_for_channel,
    )
