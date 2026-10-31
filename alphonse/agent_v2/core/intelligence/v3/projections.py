"""Structured context projections share the legacy renderer's authority boundary."""
import json

from .event_evidence import project_event_evidence
from .context_manifests import record_context_input, record_event_input


def event_evidence_projection(task, context, *, stage="projection"):
    projection = project_event_evidence(task, context)
    if projection is not None:
        serialized = json.dumps(projection, ensure_ascii=True, sort_keys=True)
        record_event_input(task, stage, projection, serialized)
        record_context_input(task, stage, serialized)
    return projection
