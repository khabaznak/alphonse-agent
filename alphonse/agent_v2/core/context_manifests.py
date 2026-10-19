"""Call-time input inventories. Raw inputs live only in the processing scope.

Manifests contain metadata, never excerpts, prompts, question text or credentials.
They follow task-checkpoint retention; rendered debug contexts are not retained.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
import json
from uuid import uuid4

_active = ContextVar("decision_manifest_recorder", default=None)


def _render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _tokens(value):
    return len(_render(value).encode("utf-8"))


@contextmanager
def manifest_scope(task, context):
    token = _active.set({"task": task, "context": context, "projections": []})
    try:
        yield
    finally:
        _active.reset(token)


def record_decisions(method):
    @wraps(method)
    def wrapped(self, task, context):
        with manifest_scope(task, context):
            return method(self, task, context)
    return wrapped


def request_text(value):
    from alphonse.agent_v2.core.context_inventory import source
    active = _active.get()
    task = active['task'] if active else None
    identity = str(task.message_id or task.task_id) if task else 'current'
    return source(value, 'request:' + identity, 'current_request')


def remember_projection(text, manifest):
    active = _active.get()
    if active is not None:
        # Ephemeral only. Match actual call inputs, never task state after the call.
        active["projections"].append((text, deepcopy(manifest)))


def projection_receipt(text):
    from alphonse.agent_v2.core.context_inventory import ContextText, receipt
    if isinstance(text, ContextText):
        return (receipt(text, purpose=text.details.get('purpose', 'default_context')),)
    active = _active.get()
    if active is not None:
        for rendered, manifest in reversed(active["projections"]):
            if text and text == rendered:
                return (deepcopy(manifest),)
    return ()


def begin_decision(*, purpose, sections, tools=(), images=(), projections=None, sources=(), omissions=()):
    active = _active.get()
    if active is None:
        return None
    from alphonse.agent_v2.core.context_inventory import section_receipts
    # Every input field gets its own inventory. Explicit prompt receipts bind
    # parts captured at composition time; Jev values preserve their inventories.
    explicit = deepcopy(list(projections or []))
    projections = section_receipts(sections, purpose=purpose)
    if explicit:
        projections = explicit + [p for p in projections if p.get('input_section') != 'prompt']
    task = active["task"]
    manifest = {
        "schema_version": 1,
        "decision_id": "decision:" + uuid4().hex,
        "purpose": purpose,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "context_builder_version": "call-input-v2",
        "provenance": "source_manifest",
        "tokenizer": "conservative_utf8_bytes",
        "sections": [{"type": name, "tokens": _tokens(value)} for name, value in sections.items()],
        "sources": deepcopy(list(sources)),
        "omissions": deepcopy(list(omissions)),
        "tokens": sum(_tokens(value) for value in sections.values()),
        "token_scope": "application sections; provider framing excluded; images uncounted",
        "tool_ids": [t.tool_id for t in tools],
        "image_count": len(images),
        "projections": projections,
        "raw_context_retained": False,
    }
    task.metadata.setdefault("decision_context_manifests", []).append(manifest)
    _publish(manifest)
    return manifest


def finish_decision(manifest, *, failed=False):
    if manifest is None:
        return
    manifest["status"] = "failed" if failed else "returned"
    _publish(manifest)


def _publish(manifest):
    active = _active.get()
    if active is not None:
        from alphonse.agent_v2.core.core import ImprovementPhase
        if active["context"].question_store is not None:
            task = active["task"]
            active["context"].question_store.save_task_checkpoint(task, status="done" if task.status == "completed" else task.status)
        purpose = manifest["purpose"]
        phase = ImprovementPhase.PLAN if "planning" in purpose or "admission" in purpose or "curation" in purpose else (
            ImprovementPhase.DO if "tactical" in purpose else ImprovementPhase.CHECK)
        active["context"].emit_activity(
            phase=phase, label="decision context recorded",
            message="Decision input manifest recorded.",
        )


def manifest_summary(manifests):
    if not manifests:
        return "Legacy work log · no call-time decision manifest was captured."
    lines = []
    for manifest in manifests:
        lines.append(f"{manifest['purpose']} · {manifest['decision_id']} · {manifest['status']} · manifest v{manifest['schema_version']}")
        lines.append(f"Builder: {manifest['context_builder_version']} · {manifest['tokens']} conservative tokens (provider framing excluded; images uncounted)")
        if manifest['provenance'] == 'legacy_aggregate_only':
            lines.append("Legacy context · aggregate call sections only; source provenance unavailable.")
        for section in manifest['sections']:
            lines.append(f"  {section['type']}: {section['tokens']} tokens")
        for source in manifest.get('sources', []):
            lines.append(f"  {source['source_id']} · {source['source_type']} · included · {source['reason']}")
        for source in manifest.get('omissions', []):
            lines.append(f"  {source['source_id']} · {source['source_type']} · omitted · {source['reason']}")
        if manifest['tool_ids']:
            lines.append("Tools: " + ", ".join(manifest['tool_ids']))
        lines.append(f"Images: {manifest['image_count']}")
        for projection in manifest['projections']:
            if 'budget' in projection:
                lines.append(f"  Projection: {projection['tokens']}/{projection['budget']} tokens · whole-record budget omission; no budget text slicing")
            else:
                lines.append(f"  Section: {projection.get('input_section', projection.get('purpose', 'context'))} · {projection['tokens']} tokens · whole-record omissions and explicit source clips")
            for budget in projection.get('budgets', []):
                lines.append(f"  Selection limits: {budget}")
            for record in projection.get('records', []):
                lines.append(f"    {record['source_id']} · {record['source_type']} · {record['status']} · {record['reason']} · {record['tokens']} tokens" + (" · source excerpt truncated" if record.get('truncated') else ""))
                if record.get('source_range'):
                    lines.append(f"      Source characters: {record['source_range']}")
                if record.get('truncated') and record.get('field_ranges'):
                    lines.append(f"      Field character ranges: {record['field_ranges']}")
            if projection.get('selected_skill_ids'):
                lines.append("  Selected skills: " + ", ".join(projection['selected_skill_ids']))
            if projection.get('loaded_skill_ids') is not None:
                lines.append('  Loaded skills: ' + ', '.join(projection['loaded_skill_ids']))
            recall = projection.get('recall', {})
            if recall:
                lines.append(f"  Recall: {recall.get('status', 'unavailable')} · {recall.get('result_count', 0)} results · further search required: {recall.get('further_search_required', False)}")
        lines.append("Raw context not retained.")
    return "\n".join(lines)
