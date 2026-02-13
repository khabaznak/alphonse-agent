from __future__ import annotations

from typing import Any, Callable


def apology_node(
    state: dict[str, Any],
    *,
    build_capability_gap_apology: Callable[..., str],
    llm_client: Any,
) -> dict[str, Any]:
    if state.get("response_text"):
        return {}
    plans = state.get("plans")
    if not isinstance(plans, list):
        return {}
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        if str(plan.get("plan_type") or "") != "CAPABILITY_GAP":
            continue
        payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
        reason = str(payload.get("reason") or "capability_gap")
        missing_slots = payload.get("missing_slots")
        apology = build_capability_gap_apology(
            state=state,
            llm_client=llm_client,
            reason=reason,
            missing_slots=missing_slots if isinstance(missing_slots, list) else None,
        )
        if apology:
            return {"response_text": apology}
        return {}
    return {}
