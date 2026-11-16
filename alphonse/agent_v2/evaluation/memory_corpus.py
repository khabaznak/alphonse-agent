"""Load and validate synthetic memory evaluation corpora."""
from pathlib import Path
from typing import Any
import json


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


