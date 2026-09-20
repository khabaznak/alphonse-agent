"""Immutable acceptance contracts with separately mutable evidence status."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


_CHECKBOX = re.compile(r"^\s*(?:\d+\s*[.):-]*|[-*])?\s*\[(?P<state>[xX ])\]\s*(?P<statement>.+?)\s*$")
_VALID_STATUSES = {"pending", "satisfied", "blocked"}


def contract_from_markdown(markdown: str, *, source_message_id: str = "") -> dict[str, Any]:
    """Create a versioned contract once from model-authored checkbox Markdown."""
    criteria: list[dict[str, Any]] = []
    for line in str(markdown or "").splitlines():
        match = _CHECKBOX.match(line)
        if match is None:
            continue
        statement = match.group("statement").strip()
        if not statement:
            continue
        criterion_id = f"ac-{len(criteria) + 1}"
        criteria.append(
            {
                "id": criterion_id,
                "statement": statement,
                "required": True,
                "verification": "semantic",
                "status": "satisfied" if match.group("state").lower() == "x" else "pending",
                "evidence_refs": [],
                "reason": "",
                "source_message_id": str(source_message_id or "").strip(),
            }
        )
    if not criteria:
        return {}
    contract = {"version": 1, "revision": 1, "criteria": criteria, "amendments": []}
    contract["definition_hash"] = definition_hash(contract)
    return contract


def normalize_contract(value: Any, *, fallback_markdown: str = "", source_message_id: str = "") -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("criteria"), list):
        return contract_from_markdown(fallback_markdown, source_message_id=source_message_id)
    normalized = deepcopy(value)
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(normalized.get("criteria") or []):
        if not isinstance(raw, dict):
            continue
        criterion_id = str(raw.get("id") or f"ac-{index + 1}").strip()
        statement = str(raw.get("statement") or "").strip()
        if not criterion_id or criterion_id in seen or not statement:
            continue
        seen.add(criterion_id)
        status = str(raw.get("status") or "pending").strip().lower()
        clean.append(
            {
                **raw,
                "id": criterion_id,
                "statement": statement,
                "required": bool(raw.get("required", True)),
                "verification": str(raw.get("verification") or "semantic").strip() or "semantic",
                "status": status if status in _VALID_STATUSES else "pending",
                "evidence_refs": _strings(raw.get("evidence_refs")),
                "reason": str(raw.get("reason") or "").strip(),
            }
        )
    normalized["version"] = max(1, _integer(normalized.get("version"), 1))
    normalized["revision"] = max(1, _integer(normalized.get("revision"), 1))
    normalized["criteria"] = clean
    normalized["amendments"] = list(normalized.get("amendments") or [])
    normalized["definition_hash"] = definition_hash(normalized)
    return normalized


def render_contract(contract: dict[str, Any]) -> str:
    rows: list[str] = []
    for index, criterion in enumerate(contract.get("criteria") or [], start=1):
        if not isinstance(criterion, dict) or criterion.get("superseded") is True:
            continue
        checked = "x" if str(criterion.get("status") or "") == "satisfied" else " "
        rows.append(f"{index}.- [{checked}] {str(criterion.get('statement') or '').strip()}")
    return "\n".join(rows) or "- (none)"


def definition_hash(contract: dict[str, Any]) -> str:
    definitions = [
        {
            "id": str(item.get("id") or ""),
            "statement": str(item.get("statement") or ""),
            "required": bool(item.get("required", True)),
            "verification": str(item.get("verification") or "semantic"),
            "superseded": bool(item.get("superseded", False)),
        }
        for item in contract.get("criteria") or []
        if isinstance(item, dict)
    ]
    payload = json.dumps(definitions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_status_patch(contract: dict[str, Any], patch: Any, *, valid_evidence_refs: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Apply status-only updates; model output can never redefine criteria."""
    updated = normalize_contract(contract)
    original_hash = updated["definition_hash"]
    rejected: list[str] = []
    updates = patch.get("updates") if isinstance(patch, dict) else None
    if not isinstance(updates, list):
        return updated, ["criteria_status_patch_invalid"]
    by_id = {str(item["id"]): item for item in updated["criteria"]}
    for raw in updates:
        if not isinstance(raw, dict):
            rejected.append("criteria_status_update_invalid")
            continue
        criterion_id = str(raw.get("criterion_id") or "").strip()
        criterion = by_id.get(criterion_id)
        if criterion is None:
            rejected.append(f"unknown_criterion:{criterion_id or '(missing)'}")
            continue
        if str(criterion.get("verification") or "semantic") == "system":
            rejected.append(f"system_criterion_model_update:{criterion_id}")
            continue
        status = str(raw.get("status") or "").strip().lower()
        if status not in _VALID_STATUSES:
            rejected.append(f"invalid_status:{criterion_id}")
            continue
        refs = _strings(raw.get("evidence_refs"))
        if status == "satisfied" and (not refs or any(ref not in valid_evidence_refs for ref in refs)):
            rejected.append(f"invalid_evidence:{criterion_id}")
            continue
        criterion["status"] = status
        criterion["evidence_refs"] = refs
        criterion["reason"] = str(raw.get("reason") or "").strip()
    if definition_hash(updated) != original_hash:
        raise ValueError("acceptance_contract_definition_mutated")
    updated["definition_hash"] = original_hash
    return updated, rejected


def apply_amendment(contract: dict[str, Any], amendment: Any, *, source_message_id: str) -> tuple[dict[str, Any], list[str]]:
    """Apply explicit steering operations while preserving an audit trail."""
    updated = normalize_contract(contract)
    rejected: list[str] = []
    operations = amendment.get("operations") if isinstance(amendment, dict) else None
    if not isinstance(operations, list):
        return updated, ["criteria_amendment_invalid"]
    by_id = {str(item["id"]): item for item in updated["criteria"]}
    next_index = len(updated["criteria"]) + 1
    accepted_operations: list[dict[str, Any]] = []
    for raw in operations:
        if not isinstance(raw, dict):
            rejected.append("criteria_amendment_operation_invalid")
            continue
        operation = str(raw.get("operation") or "").strip().lower()
        if operation == "add":
            statement = str(raw.get("statement") or "").strip()
            if not statement:
                rejected.append("criteria_add_statement_missing")
                continue
            criterion_id = f"ac-{next_index}"
            next_index += 1
            item = {
                "id": criterion_id,
                "statement": statement,
                "required": True,
                "verification": "semantic",
                "status": "pending",
                "evidence_refs": [],
                "reason": "",
                "source_message_id": source_message_id,
            }
            updated["criteria"].append(item)
            by_id[criterion_id] = item
            accepted_operations.append({"operation": "add", "criterion_id": criterion_id, "statement": statement})
            continue
        if operation == "supersede":
            criterion_id = str(raw.get("criterion_id") or "").strip()
            replacement = str(raw.get("statement") or "").strip()
            existing = by_id.get(criterion_id)
            if existing is None or not replacement:
                rejected.append(f"criteria_supersede_invalid:{criterion_id or '(missing)'}")
                continue
            existing["superseded"] = True
            replacement_id = f"ac-{next_index}"
            next_index += 1
            item = {
                "id": replacement_id,
                "statement": replacement,
                "required": True,
                "verification": "semantic",
                "status": "pending",
                "evidence_refs": [],
                "reason": "",
                "source_message_id": source_message_id,
                "supersedes": criterion_id,
            }
            updated["criteria"].append(item)
            by_id[replacement_id] = item
            accepted_operations.append({"operation": "supersede", "criterion_id": criterion_id, "replacement_id": replacement_id})
            continue
        rejected.append(f"criteria_amendment_operation_unsupported:{operation or '(missing)'}")
    if accepted_operations:
        updated["revision"] = int(updated.get("revision") or 1) + 1
        updated["amendments"].append(
            {
                "revision": updated["revision"],
                "source_message_id": str(source_message_id or "").strip(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "operations": accepted_operations,
            }
        )
    updated["definition_hash"] = definition_hash(updated)
    return updated, rejected


def required_criteria_complete(contract: dict[str, Any]) -> bool:
    active = [item for item in contract.get("criteria") or [] if isinstance(item, dict) and item.get("superseded") is not True and bool(item.get("required", True))]
    return bool(active) and all(str(item.get("status") or "") == "satisfied" for item in active)


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
