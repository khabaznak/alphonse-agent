from __future__ import annotations

import json
from alphonse.agent.observability.log_manager import get_component_logger
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import (
    CAPABILITY_GAP_REFLECTION_SYSTEM_PROMPT,
    CAPABILITY_GAP_REFLECTION_USER_TEMPLATE,
    render_prompt_template,
)
from alphonse.agent.cognition.capability_gaps.triage import (
    detect_language,
    triage_gap_text,
)
from alphonse.agent.cognition.providers.factory import build_llm_client
from alphonse.agent.nervous_system.capability_gaps import list_gaps
from alphonse.agent.nervous_system.gap_proposals import (
    get_pending_proposal_for_gap,
    insert_gap_proposal,
)

logger = get_component_logger("cognition.capability_gaps.reflection")


def reflect_gaps(limit: int = 50) -> list[str]:
    llm_client = _safe_build_llm()
    created: list[str] = []
    gaps = list_gaps(status="open", limit=limit, include_all=False)
    for gap in gaps:
        gap_id = gap.get("gap_id")
        if not gap_id:
            continue
        if get_pending_proposal_for_gap(str(gap_id)):
            continue
        proposal = _build_proposal(gap, llm_client)
        proposal_id = insert_gap_proposal(proposal)
        created.append(proposal_id)
    return created


def _build_proposal(gap: dict[str, Any], llm_client: Any | None) -> dict[str, Any]:
    text = str(gap.get("user_text") or "").strip()
    language = detect_language(text)
    triage = triage_gap_text(text) if text else {"category": "intent_missing", "confidence": 0.2}
    if triage.get("suggested_intent"):
        return _proposal_from_triage(gap, triage, language, notes="deterministic")

    llm = _llm_proposal(text, language, llm_client) if text else None
    if llm:
        return _proposal_from_llm(gap, llm, language)
    return _proposal_from_triage(
        gap,
        {
            "category": triage.get("category") or "intent_missing",
            "suggested_intent": None,
            "confidence": 0.2,
        },
        language,
        notes="llm_unavailable",
    )


def _proposal_from_triage(
    gap: dict[str, Any], triage: dict[str, Any], language: str, notes: str
) -> dict[str, Any]:
    confidence = float(triage.get("confidence") or 0.2)
    next_action = "investigate" if confidence < 0.7 else "plan"
    return {
        "gap_id": gap.get("gap_id"),
        "created_at": _now_iso(),
        "status": "pending",
        "proposed_category": triage.get("category") or "intent_missing",
        "confidence": confidence,
        "proposed_next_action": next_action,
        "proposed_intent_name": triage.get("suggested_intent"),
        "proposed_clarifying_question": None,
        "notes": notes,
        "language": language,
        "reviewer": None,
        "reviewed_at": None,
    }


def _proposal_from_llm(
    gap: dict[str, Any], proposal: dict[str, Any], language: str
) -> dict[str, Any]:
    confidence = float(proposal.get("confidence") or 0.0)
    next_action = proposal.get("proposed_next_action") or "investigate"
    if confidence < 0.7 and next_action in {"fix_now", "plan"}:
        next_action = "investigate"
    return {
        "gap_id": gap.get("gap_id"),
        "created_at": _now_iso(),
        "status": "pending",
        "proposed_category": proposal.get("proposed_category") or "intent_missing",
        "confidence": confidence,
        "proposed_next_action": next_action,
        "proposed_intent_name": proposal.get("proposed_intent_name"),
        "proposed_clarifying_question": proposal.get("proposed_clarifying_question"),
        "notes": proposal.get("notes"),
        "language": language,
        "reviewer": None,
        "reviewed_at": None,
    }


def _llm_proposal(
    text: str, language: str, llm_client: Any | None
) -> dict[str, Any] | None:
    if not llm_client or not text:
        return None
    prompt = render_prompt_template(
        CAPABILITY_GAP_REFLECTION_USER_TEMPLATE,
        {
            "LANGUAGE": language,
            "USER_TEXT": text,
        },
    )
    try:
        raw = llm_client.complete(
            system_prompt=CAPABILITY_GAP_REFLECTION_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
    except Exception as exc:
        logger.warning("Gap reflection LLM failed: %s", exc)
        return None
    data = _parse_json(str(raw))
    if not isinstance(data, dict):
        return None
    return data


def _parse_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_build_llm() -> Any | None:
    try:
        return build_llm_client()
    except Exception as exc:
        logger.warning("Gap reflection LLM unavailable: %s", exc)
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
