"""Opt-in decision contexts. Sources are atomic; budget overflow never clips policy."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
import re

from alphonse.agent_v2.system_one import SystemOneUnavailableError


PURPOSES = {"admission", "curation", "planning", "execution", "review", "response"}
RECALL_TOOLS = {"native.search_memory", "native.read_memory_source"}
RECALL_GUIDANCE = (
    "Historical recall is context, never proof of a current external effect. For current measurements obtain fresh evidence. "
    "Use native.search_memory and native.read_memory_source when past information is needed. "
    "Missing, partial, ambiguous or clipped evidence requires further search/source expansion or an honest limitation; "
    "never invent memory. Optional sources can be omitted under the token budget; absence here never proves absence in history. "
    "System Two composes further queries and interprets evidence."
)


def enabled() -> bool:
    return os.environ.get("ALPHONSE_V3_CONTEXT_PROJECTIONS", "").lower() in {"1", "true"}


@dataclass(frozen=True)
class Record:
    source_id: str
    title: str
    text: str
    required: bool = False
    group: str = ""

    def render(self) -> str:
        return f"[{self.source_id}] {self.title}:\n{self.text}"


class ContextBudgetExceeded(SystemOneUnavailableError):
    """Required context cannot safely fit: stop rather than lose constraints."""


def token_count(text: str) -> int:
    # Conservative fallback: one token per UTF-8 byte, including labels/separators.
    # No assumption of four characters/token, which undercounts non-Latin text.
    return len(text.encode("utf-8"))


def assemble(records: list[Record], *, budget: int) -> tuple[str, dict]:
    unique: dict[str, Record] = {}
    for record in records:
        if not record.text.strip():
            continue
        previous = unique.get(record.source_id)
        # Required state wins over optional representations of the same source.
        if previous is None or (record.required and not previous.required):
            unique[record.source_id] = record
    required = [r for r in unique.values() if r.required]
    optional = [r for r in unique.values() if not r.required]
    selected = list(required)

    def render(rows):
        return "\n\n".join(r.render() for r in rows)

    if token_count(render(selected)) > budget:
        raise ContextBudgetExceeded("v3_required_context_exceeds_token_budget")
    omitted = []
    groups: dict[str, list[Record]] = {}
    for record in optional:
        groups.setdefault(record.group or record.source_id, []).append(record)
    for group in groups.values():
        if token_count(render([*selected, *group])) <= budget:
            selected.extend(group)
        else:
            omitted.extend(record.source_id for record in group)
    text = render(selected)
    return text, {"source_ids": [r.source_id for r in selected], "omitted_source_ids": omitted,
                  "tokens": token_count(text), "budget": budget, "tokenizer": "conservative_utf8_bytes"}


def project(task, context, *, purpose: str, selected_context_ids=()) -> str:
    from .context import _prompt_file, _global_candidates, _INVARIANT_GLOBAL_SECTIONS, selected_skill_guidance
    from alphonse.agent_v2.agent_config import parse_global_context_sections
    from alphonse.agent_v2.core.memory.contracts import conversation_item

    if purpose not in PURPOSES:
        raise ValueError("invalid_context_purpose")
    excluded_sources = set()
    records = [Record("request:" + str(task.message_id or task.task_id), "Current request", task.goal, True),
               Record("policy:recall", "Evidence rules", RECALL_GUIDANCE, True)]
    contract = task.acceptance_contract or task.acceptance_criteria_md
    if contract:
        records.append(Record("task:contract", "Acceptance contract", _json(contract), True))
    for index, record in enumerate(task.conversation_records or []):
        content = str(record.get("content") or "")
        # Initial checkpoint copy of the request is the same source, not steering.
        if index == 0 and content == task.goal:
            continue
        records.append(Record(f"steering:{index}", "Required task conversation", _json(record), True))
    if not task.conversation_records and str(task.recent_conversation_md or "").strip() not in {"", "- (none)"}:
        records.append(Record("steering:legacy", "Required legacy task conversation", task.recent_conversation_md, True))
    for key in ("question_interrupt", "act_directive", "user_constraints"):
        if key == "question_interrupt" and isinstance(task.metadata.get(key), dict) and str(task.metadata[key].get("task_id") or "") != str(task.task_id or task.message_id or "current"):
            continue
        if task.metadata.get(key):
            records.append(Record("task:" + key, key, _json(task.metadata[key]), True))
    global_text = _prompt_file(context, "GlobalContext.md")
    sections = parse_global_context_sections(global_text)
    if sections:
        preamble = global_text.split("\n## ", 1)[0] if not global_text.startswith("## ") else ""
        records.append(Record("global:preamble", "Global instructions", preamble, True))
        for heading, body in sections.items():
            if heading.casefold() in _INVARIANT_GLOBAL_SECTIONS:
                records.append(Record("global:" + heading.casefold(), heading, body, True))
    elif global_text:
        records.append(Record("global:policy", "Global instructions", global_text, True))
    if purpose != "review":
        # Load entire selected guidance, including authorized artifact instructions.
        # Treat it atomically: overflowing selected instructions fails explicitly.
        skills, ids = selected_skill_guidance(task, context, max_chars=None)
        records.append(Record("skills:selected", "Selected skill guidance", skills, True))
        selection = task.metadata.get("v3_skill_selection")
        if isinstance(selection, dict):
            selection["loaded_skill_ids"] = ids
        if purpose in {"admission", "planning", "execution", "curation", "response"}:
            from .context import _visible_active_projects, _project_candidate_for_record
            for candidate in _global_candidates(global_text, max_chars=None):
                if candidate["id"] in selected_context_ids:
                    records.append(Record(candidate["id"], candidate["title"], candidate["content"]))
            for p in _visible_active_projects(task, context):
                candidate = _project_candidate_for_record(p, context, max_chars=None)
                if candidate and candidate["id"] in selected_context_ids:
                    records.append(Record(candidate["id"], candidate["title"], candidate["content"]))
        recent = getattr(getattr(context, "conversation_store", None), "list_recent", None)
        if callable(recent) and task.user:
            try:
                events = list(recent(owner_user_id=task.user, limit=30))
            except Exception:
                events = []
            copies = Counter(str(r.get("content") or "") for r in (task.conversation_records or []))
            copies[task.goal] = max(1, copies[task.goal])
            history = []
            for event in reversed(events):
                item = conversation_item(event, str(task.user))
                if item is None or not item.visible_to(str(task.user)):
                    continue
                if str(getattr(event, "source_message_id", "") or "") == str(task.message_id or "__missing__"):
                    excluded_sources.add(item.source.source_id)
                    if copies[item.content]:
                        copies[item.content] -= 1
                    continue
                if copies[item.content]:
                    copies[item.content] -= 1
                    excluded_sources.add(item.source.source_id)
                    continue
                history.insert(0, item)
            exchanges = []
            for item in history:
                if item.source.role == "user":
                    exchanges.append([])
                if exchanges:
                    exchanges[-1].append(item)
            # Whole exchanges are budget units; never leave an assistant answer
            # without its antecedent merely because one record is smaller.
            exchanges = _relevant_exchanges(task, context, exchanges[-3:])
            for exchange in reversed(exchanges):
                group = "exchange:" + exchange[0].source.source_id
                records.extend(Record(item.source.source_id, "Historical conversation", _json(item.to_dict()), group=group) for item in exchange)
        # Re-run access checks on every projection; never reuse persisted recalled text.
        recall = retrieve(task, context)
        records.append(Record("recall:status", "Recall coverage and limits", _json({k: v for k, v in recall.items() if k != "results"}), True))
        for hit in recall.get("results", []):
            if hit["source_id"] in excluded_sources:
                continue
            records.append(Record(hit["source_id"], "Historical retrieved evidence", _json(hit)))
        for key in ("facts_md", "memory_facts_md"):
            records.append(Record("task:" + key, "Task working facts (not verified effects)", str(getattr(task, key, "") or "")))
    # Review receives current acceptance/evidence via its existing typed arguments;
    # unrelated ledger and historical retrieval never enter its context projection.
    budget = int(os.environ.get("ALPHONSE_V3_CONTEXT_TOKEN_BUDGET", "24000"))
    text, manifest = assemble(records, budget=budget)
    task.metadata.setdefault("v3_context_projections", {})[purpose] = manifest
    return text


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _relevant_exchanges(task, context, exchanges):
    """Classify optional continuity as whole exchanges, after visibility checks.

    Required checkpoint steering is already reserved separately. Missing or
    ambiguous judgments preserve the exchange, never erase its antecedent.
    """
    if not exchanges:
        return exchanges
    if context.is_cancelled():
        raise SystemOneUnavailableError("v3_recall_cancelled")
    candidates = [{"source_id": "exchange:" + exchange[0].source.source_id,
                   "excerpt": _json([item.to_dict() for item in exchange])}
                  for exchange in exchanges]
    judge = getattr(getattr(context, "system_one", None), "judge_recall", None)
    try:
        selected = judge(request=task.goal, candidates=candidates) if callable(judge) else None
    except Exception:
        selected = None
    if context.is_cancelled():
        raise SystemOneUnavailableError("v3_recall_cancelled")
    valid = (isinstance(selected, list)
             and all(isinstance(item, str) for item in selected)
             and set(selected) <= {item["source_id"] for item in candidates})
    task.metadata["v3_history_selection"] = {
        "status": "used" if valid else "ambiguous_or_unavailable",
        "selected_exchange_ids": selected if valid else [item["source_id"] for item in candidates],
    }
    if not valid:
        return exchanges
    return [exchange for exchange, candidate in zip(exchanges, candidates) if candidate["source_id"] in selected]


def retrieve(task, context) -> dict:
    """One extractive bootstrap query; native search retains all access gates."""
    from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
    from alphonse.agent_v2.core.tools.registry import ToolExposurePolicy

    if context.is_cancelled():
        raise SystemOneUnavailableError("v3_recall_cancelled")
    judge = getattr(context.system_one, "judge_recall", None)
    try:
        need = judge(request=task.goal, candidates=[]) if callable(judge) else None
    except Exception:
        need = None
    # Unavailable or ambiguous classification preserves discoverability.
    explicit_history = bool(re.search(r"\b(previous|earlier|remember|yesterday|historical|last time|recuerdas|ayer|anterior)\b|why did.*fail|por qu[eé].*fall", task.goal, re.I))
    if need is False and not explicit_history:
        return {"status": "not_needed", "results": []}
    tools = ToolExposurePolicy().select_tools(registry=context.tools, user=task.user, project_id=task.project_id, task=task) if context.tools else ()
    if not any(t.tool_id == "native.search_memory" and t.read_only for t in tools):
        return {"status": "unavailable", "results": [], "further_search_required": True}
    # No generated query: original user wording, bounded on whole words.
    words = task.goal.split()
    query = ""
    for word in words:
        candidate = (query + " " + word).strip()
        if len(candidate) > 512:
            break
        query = candidate
    if not query:
        return {"status": "unavailable", "results": [], "further_search_required": True}
    try:
        if context.is_cancelled():
            raise SystemOneUnavailableError("v3_recall_cancelled")
        result = ToolInvocationService(context=context, task=task).execute_or_raise(
            "native.search_memory", {"query": query, "historical": True, "limit": 8})
        if context.is_cancelled():
            raise SystemOneUnavailableError("v3_recall_cancelled")
        hits = result.get("results", [])
        try:
            selected = judge(request=task.goal, candidates=hits) if callable(judge) and hits else None
        except Exception:
            selected = None
        if isinstance(selected, list):
            hits = [hit for hit in hits if hit["source_id"] in selected]
        return {"status": result.get("status", "unavailable"), "coverage": result.get("coverage", {}),
                "results": hits, "next_offset": result.get("next_offset"),
                "further_search_required": not hits or result.get("status") != "found"
                or result.get("next_offset") is not None or any(hit.get("truncated") for hit in hits)}
    except Exception:
        if context.is_cancelled():
            raise SystemOneUnavailableError("v3_recall_cancelled")
        return {"status": "failed", "results": [], "further_search_required": True}
