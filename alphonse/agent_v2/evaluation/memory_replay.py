"""Synthetic context baseline. No runtime configuration or providers are loaded offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
from alphonse.agent_v2.core.telemetry import approximate_tokens
from alphonse.agent_v2.core.tools.registry.native.memory import execute_search_memory


def load_corpus(path: Path) -> dict[str, Any]:
    corpus = json.loads(path.read_text())
    if corpus.get("schema_version") != 1 or corpus.get("synthetic") is not True:
        raise ValueError("versioned_synthetic_corpus_required")
    case_ids = set()
    for case in corpus["cases"]:
        if case["id"] in case_ids:
            raise ValueError("duplicate_case_id")
        case_ids.add(case["id"])
        ids = [s["id"] for s in case["sources"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_source_id")
        if len({s["content"] for s in case["sources"]}) != len(ids):
            raise ValueError("duplicate_source_content")
        expected = case["expected"]
        for key in ("necessary_sources", "forbidden_sources"):
            if not set(expected[key]) <= set(ids):
                raise ValueError("unknown_expected_source")
        if set(expected["necessary_sources"]) & set(expected["forbidden_sources"]):
            raise ValueError("conflicting_source_expectations")
        for source in case["sources"]:
            if not source["content"] or source["visibility"] not in {"owner", "private", "shared"}:
                raise ValueError("invalid_source")
        # These fields are mandatory, even when empty.
        for key in ("forbidden_actions", "required_answer_fragments", "forbidden_answer_fragments", "allowed_historical_claims"):
            expected[key]
    return corpus


def assemble(case: dict[str, Any]) -> dict[str, Any]:
    """Exercise current SQLite search, native recall and real V3 context assembly.

    Shared records are copied into the requesting owner's view by the fixture;
    today's conversation store has no sharing API. No production data is read.
    """
    store = SQLiteConversationStore(":memory:")
    try:
        for s in case["sources"]:
            store.record(owner_user_id=case["user"] if s["visibility"] == "shared" else s["owner"],
                         project_id=s.get("project", ""), role=s.get("role", "user"),
                         content=s["content"], source="synthetic", source_message_id=s["id"],
                         created_at=s["created_at"])
        search_calls = []

        class SearchView:
            def list_recent(self, **kwargs):
                return store.list_recent(**kwargs)

            def recall_items(self, requester):
                mode = case.get("search_mode", "available")
                search_calls.append(mode)
                if mode == "unavailable":
                    raise OSError("synthetic_search_unavailable")
                if mode == "empty":
                    return []
                return store.recall_items(requester)

            def search(self, **kwargs):
                mode = case.get("search_mode", "available")
                search_calls.append(mode)
                if mode == "unavailable":
                    raise RuntimeError("synthetic_search_unavailable")
                if mode == "empty":
                    return []
                return store.search(**kwargs)

        class EmptyLedger:
            def recall_items(self, *, user_id, project_id):
                return []

            def search(self, **kwargs):
                return ""

        task = TaskState(user=case["user"], goal=case["message"])
        for turn in case.get("active_turns", []):
            task.append_conversation_message(turn["speaker"], turn["content"])
        task.append_conversation_message(case["user"], case["message"])
        view = SearchView()
        context = SimpleNamespace(conversation_store=view, project_store=None, prompts=None)
        prompt = task_context_for_decision(task, context)
        recall = ""
        error = ""
        if case.get("query"):
            try:
                result = execute_search_memory({"query": case["query"]}, context=SimpleNamespace(
                    task=task, memory=EmptyLedger(), conversation_store=view))
                recall = result["matches_markdown"]
            except RuntimeError as exc:
                error = str(exc)
        combined = prompt + ("\n\nRecall results:\n" + recall if recall else "")
        # Exact fixture text identifies sources without adding artificial tags to
        # production prompts. Long/truncated sources need a future runtime manifest.
        included = [s["id"] for s in case["sources"] if s["content"] in combined]
        retrieved = [s["id"] for s in case["sources"] if s["content"] in recall]
        return {"prompt": combined, "included_sources": included, "retrieved_sources": retrieved,
                "search_backend_states": search_calls, "search_error": error,
                "tool_calls": int(bool(case.get("query"))), "prompt_tokens_estimate": approximate_tokens(combined)}
    finally:
        store._memory.close()


def score_answer(case: dict[str, Any], trace: dict[str, Any], included: list[str], *, rules: dict[str, Any]) -> dict[str, Any]:
    from alphonse.agent_v2.evaluation.memory_scoring import score_response

    return score_response(case, trace, included, rules)


def run(corpus: dict[str, Any], *, responder=None, timing: bool = False, rubric: dict[str, Any] | None = None) -> dict[str, Any]:
    if responder is not None and rubric is None:
        raise ValueError("response_rubric_required")
    rows = []
    for case in corpus["cases"]:
        started = perf_counter()
        observed = assemble(case)
        required = set(case["expected"]["necessary_sources"])
        included = set(observed["included_sources"])
        retrieval = {"necessary_source_recall": len(required & included) / len(required) if required else 1.0,
                     "search_source_recall": len(required & set(observed["retrieved_sources"])) / len(required) if required else None,
                     "irrelevant_sources": sorted(included - required),
                     "forbidden_sources": sorted(included & set(case["expected"]["forbidden_sources"])),
                     "private_sources": sorted(s["id"] for s in case["sources"] if s["id"] in included
                                               and s["visibility"] == "private" and s["owner"] != case["user"])}
        prompt = observed.pop("prompt")
        row = {"id": case["id"], **observed, "retrieval": retrieval,
               "answer_quality": None, "model_calls": 0, "latency_ms": None}
        if responder is not None:
            # Deliberately exclude fixture expectations, forbidden sources, and
            # the corpus from the provider input. Returned actions are never run.
            trace = responder({"message": case["message"], "context": prompt,
                               "source_ids": observed["included_sources"],
                               "response_contract": {"answer": "text", "claims": [{"text": "exact historical assertion quoted from context; list every historical assertion in answer", "source_ids": ["source identifier"]}], "actions": ["proposed action identifier"]}})
            row["answer_quality"] = score_answer(case, trace, observed["included_sources"], rules=rubric["cases"][case["id"]])
            row["response"] = trace
            row["model_calls"] = 1
        if timing:
            row["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        rows.append(row)
    encoded = json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode()
    report = {"schema_version": 1, "corpus_version": corpus["version"],
            "corpus_sha256": hashlib.sha256(encoded).hexdigest(),
            "adapter": "current-v3-context-and-native-search-v3",
            "mode": "live-response" if responder else "offline-context-only", "cases": rows}
    if responder is not None:
        from alphonse.agent_v2.evaluation.memory_scoring import content_hash
        report.update(scorer_version=rubric["version"], rubric_sha256=content_hash(rubric))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    corpus = load_corpus(args.corpus)
    report = run(corpus, timing=args.timing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
