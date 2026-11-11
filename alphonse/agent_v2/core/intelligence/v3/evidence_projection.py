"""Decision-sized, source-linked evidence without a new evidence store.

The caller supplies durable source references, an actor-scoped access check, and
an existing authorized source loader. Sources must be atomic visibility units;
never wrap mixed-audience data in a single source. Project immediately before
provider use and re-run the projection after any authority or evidence change.
This module does not itself authorize actors or make provider calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Iterable

from .projections import Record, assemble

# Smaller is selected first. Mandatory sources always precede optional sources.
_KIND_ORDER = {
    "goal": 0,
    "constraint": 1,
    "effect": 2,
    "current_result": 3,
    "observation": 4,
}
_REQUIRED_KINDS = frozenset({"goal", "constraint", "effect", "current_result"})
_COVERAGE_TEXT = (
    "Coverage applies only to selected sources, not all relevant evidence. "
    "Missing, omitted, clipped or unavailable evidence proves neither absence "
    "nor outcome. Expand authorized sources as needed."
)


@dataclass(frozen=True)
class EvidenceSource:
    """Reference to one actor-authorized, indivisible source.

    `truncated` describes clipping by the producer, not this projection;
    incomplete evidence must not be used as if it were a complete observation.
    `required` may make additional observations mandatory; goal, constraints,
    effect state and current results are always mandatory when supplied.
    """

    source_id: str
    kind: str
    required: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class EvidenceProjection:
    text: str
    manifest: dict


class EvidenceAccessError(PermissionError):
    """A required source could not be authorized or safely loaded."""


def project_evidence(
    sources: Iterable[EvidenceSource], *, actor: str, budget: int,
    authorize: Callable[[str, str], bool],
    load: Callable[[str, str], str],
) -> EvidenceProjection:
    """Project whole accessible sources within a conservative UTF-8-byte budget.

    The manifest contains source IDs and coverage metadata, never source content.
    Inaccessible optional source identifiers are not disclosed. The loader must
    return a source at the same actor-scoped authorization boundary and must not
    include credentials or material from other visibility units. An authorization
    callback failure is treated as denial. Required denial or loader failure stops
    the projection. Expansion uses this same function with requested source refs,
    so access is checked again rather than trusting an earlier manifest.
    """
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("evidence_actor_required")
    if type(budget) is not int or budget <= 0:
        raise ValueError("evidence_budget_invalid")
    rows = list(sources)
    if any(not isinstance(row, EvidenceSource) or not isinstance(row.source_id, str)
           or not row.source_id.strip() or row.source_id == "projection:coverage"
           or row.kind not in _KIND_ORDER for row in rows):
        raise ValueError("evidence_source_invalid")
    if len({row.source_id for row in rows}) != len(rows):
        raise ValueError("evidence_source_duplicate")
    if not any(row.kind == "goal" for row in rows):
        raise ValueError("evidence_goal_required")
    records = [Record("projection:coverage", "Evidence coverage", _COVERAGE_TEXT, True)]
    loaded: dict[str, EvidenceSource] = {}
    denied_count = 0
    unavailable_count = 0

    def allowed(row: EvidenceSource) -> bool:
        try:
            return authorize(actor, row.source_id) is True
        except Exception:
            return False

    for row in sorted(rows, key=lambda value: _KIND_ORDER[value.kind]):
        required = row.required or row.kind in _REQUIRED_KINDS
        if not allowed(row):
            if required:
                raise EvidenceAccessError("required_evidence_denied")
            denied_count += 1
            continue
        try:
            content = load(actor, row.source_id)
            if not isinstance(content, str) or not content.strip():
                raise ValueError("evidence_empty")
        except Exception:
            if required:
                raise EvidenceAccessError("required_evidence_unavailable") from None
            unavailable_count += 1
            continue
        # A source can be revoked while it is loaded. Do not return its contents.
        if not allowed(row):
            if required:
                raise EvidenceAccessError("required_evidence_denied")
            denied_count += 1
            continue
        if row.truncated and required:
            raise EvidenceAccessError("required_evidence_incomplete")
        if row.truncated:
            content = "[Producer marked this source as truncated.]\n" + content
        records.append(Record(row.source_id, row.kind.replace("_", " ").title(), content, required))
        loaded[row.source_id] = row

    # Revalidate after all loading: later loads may have changed access to an
    # earlier source. The caller must still fence authorization at provider use.
    for source_id, row in list(loaded.items()):
        if not allowed(row):
            if row.required or row.kind in _REQUIRED_KINDS:
                raise EvidenceAccessError("required_evidence_denied")
            denied_count += 1
            del loaded[source_id]
            records = [record for record in records if record.source_id != source_id]

    # Coverage is sent to the decision maker as a required source, not merely
    # attached to telemetry. Account for its exact bytes in the same budget.
    # Omitted authorized IDs permit targeted expansion. Withheld IDs never do.
    coverage = None
    seen = set()
    while True:
        details = "" if coverage is None else "\n" + json.dumps(coverage, ensure_ascii=True, sort_keys=True)
        records[0] = Record("projection:coverage", "Evidence coverage", _COVERAGE_TEXT + details, True)
        text, manifest = assemble(records, budget=budget)
        next_coverage = {
            "denied_count": denied_count,
            "unavailable_count": unavailable_count,
            "over_budget_source_ids": manifest["omitted_source_ids"],
            "producer_truncated_source_ids": [source_id for source_id, row in loaded.items() if row.truncated],
        }
        next_coverage["complete"] = not (denied_count or unavailable_count
            or next_coverage["over_budget_source_ids"] or next_coverage["producer_truncated_source_ids"])
        if coverage == next_coverage:
            break
        fingerprint = json.dumps(next_coverage, sort_keys=True)
        if fingerprint in seen:
            # A varying coverage size must never result in an unbounded cycle.
            from .projections import ContextBudgetExceeded
            raise ContextBudgetExceeded("v3_evidence_coverage_does_not_stabilize")
        seen.add(fingerprint)
        coverage = next_coverage
    for record in manifest["records"]:
        item = loaded.get(record["source_id"])
        if item is not None:
            record["source_type"] = item.kind
            record["truncated"] = item.truncated
    manifest.update({
        "schema_version": 1,
        "purpose": "bounded_evidence",
        "coverage": {
            "denied_count": denied_count,
            "unavailable_count": unavailable_count,
            # Denied/unavailable identities remain private, but omissions are
            # still visible as counts and cannot masquerade as complete recall.
            "omitted_count": denied_count + unavailable_count + len(manifest["omitted_source_ids"]),
            "over_budget_count": len(manifest["omitted_source_ids"]),
            "truncated_count": sum(row.truncated for row in loaded.values()),
            "absence_is_not_evidence": True,
            "complete": not (denied_count or unavailable_count or manifest["omitted_source_ids"]
                             or any(row.truncated for row in loaded.values())),
        },
        "raw_context_retained": False,
    })
    # Reuse the existing call-time manifest mechanism; it retains metadata only.
    from alphonse.agent_v2.core.context_manifests import remember_projection
    remember_projection(text, manifest)
    return EvidenceProjection(text, manifest)
