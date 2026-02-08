from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Any

from alphonse.agent.nervous_system.capability_gaps import list_gaps
from alphonse.agent.nervous_system.gap_proposals import (
    insert_gap_proposal,
    list_gap_proposals,
)


def coalesce_open_intent_gaps(
    *,
    limit: int = 300,
    min_cluster_size: int = 2,
) -> list[str]:
    gaps = list_gaps(status="open", limit=limit, include_all=False)
    if not gaps:
        return []

    clusters: dict[str, dict[str, Any]] = defaultdict(_new_cluster)
    for gap in gaps:
        metadata = gap.get("metadata") if isinstance(gap.get("metadata"), dict) else {}
        proposed_intent = _normalize_intent_name(metadata.get("proposed_intent"))
        if not proposed_intent:
            continue
        cluster = clusters[proposed_intent]
        cluster["count"] += 1
        cluster["gap_ids"].append(str(gap.get("gap_id") or ""))
        cluster["latest_gap_id"] = cluster["latest_gap_id"] or str(gap.get("gap_id") or "")
        cluster["examples"] = _append_unique(
            cluster["examples"], str(gap.get("user_text") or ""), max_items=3
        )
        aliases = metadata.get("proposed_intent_aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                normalized_alias = str(alias or "").strip().lower()
                if normalized_alias:
                    cluster["aliases"].add(normalized_alias)
        confidence = metadata.get("proposed_intent_confidence")
        if isinstance(confidence, (int, float)):
            cluster["confidences"].append(float(confidence))

    pending = list_gap_proposals(status="pending", limit=1000)
    pending_intents = {
        _normalize_intent_name(item.get("proposed_intent_name")) for item in pending
    }
    pending_intents.discard(None)

    created: list[str] = []
    for intent_name, cluster in clusters.items():
        if cluster["count"] < min_cluster_size:
            continue
        if intent_name in pending_intents:
            continue

        proposal = {
            "gap_id": cluster["latest_gap_id"],
            "status": "pending",
            "proposed_category": "intent_missing",
            "confidence": _cluster_confidence(cluster),
            "proposed_next_action": "plan",
            "proposed_intent_name": intent_name,
            "proposed_clarifying_question": None,
            "notes": _cluster_notes(intent_name, cluster),
            "language": _cluster_language(cluster),
            "reviewer": None,
            "reviewed_at": None,
        }
        created.append(insert_gap_proposal(proposal))
    return created


def _new_cluster() -> dict[str, Any]:
    return {
        "count": 0,
        "gap_ids": [],
        "latest_gap_id": None,
        "examples": [],
        "aliases": set(),
        "confidences": [],
    }


def _normalize_intent_name(value: object | None) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return text.replace(" ", "_")


def _append_unique(items: list[str], value: str, *, max_items: int) -> list[str]:
    text = value.strip()
    if not text:
        return items
    if text not in items and len(items) < max_items:
        items.append(text)
    return items


def _cluster_confidence(cluster: dict[str, Any]) -> float:
    confidences = cluster.get("confidences") or []
    if not confidences:
        return 0.5
    return round(mean(confidences), 3)


def _cluster_language(cluster: dict[str, Any]) -> str:
    examples = [str(item or "").lower() for item in cluster.get("examples") or []]
    if any("¿" in item or "qué" in item or "hora" in item for item in examples):
        return "es"
    return "en"


def _cluster_notes(intent_name: str, cluster: dict[str, Any]) -> str:
    payload = {
        "source": "coalesced_open_gaps",
        "intent_candidate": intent_name,
        "count": int(cluster.get("count") or 0),
        "aliases": sorted(cluster.get("aliases") or []),
        "gap_ids": [item for item in cluster.get("gap_ids") or [] if item],
        "examples": cluster.get("examples") or [],
    }
    return json.dumps(payload, ensure_ascii=True)
