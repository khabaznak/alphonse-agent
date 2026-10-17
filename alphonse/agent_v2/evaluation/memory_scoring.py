"""Versioned, deterministic response checks for the synthetic memory corpus.

These are bounded fixture checks, not a semantic judge or execution verifier.
Source attribution and forbidden-context inclusion are independent measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

SCORER_VERSION = "memory-response-v2"


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_rubric(path: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    rubric = json.loads(path.read_text())
    if rubric.get("version") != SCORER_VERSION or rubric.get("corpus_version") != corpus["version"]:
        raise ValueError("response_rubric_version_mismatch")
    if set(rubric["cases"]) != {case["id"] for case in corpus["cases"]}:
        raise ValueError("response_rubric_case_mismatch")
    for rules in rubric["cases"].values():
        if not rules["required"]:
            raise ValueError("response_requirements_required")
        for patterns in rules["required"]:
            if not patterns:
                raise ValueError("empty_response_requirement")
            for pattern in patterns:
                re.compile(pattern)
        for key in ("forbidden", "negation_exceptions"):
            for pattern in rules[key]:
                re.compile(pattern)
    return rubric


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    # Keep exact spelling of identifiers containing digits or punctuation.
    return [word if re.search(r"[\d_.-]", word) else word.casefold()
            for word in re.findall(r"\w+(?:[._'-]\w+)*", _normalize(text))]


def _quote_matches(text: str, source: str) -> bool:
    text = re.sub(r"^(?:User|Assistant):\s*", "", _normalize(text), flags=re.I)
    quote, original = _tokens(text), _tokens(source)
    if not quote or (len(quote) < 3 and quote != original):
        return False
    for start in range(len(original) - len(quote) + 1):
        end = start + len(quote)
        if original[start:end] != quote:
            continue
        omitted = original[:start] + original[end:]
        # A substring can invert a source by omitting a negation, uncertainty,
        # correction or failure qualifier. Do not certify those partial quotes.
        if any(word.casefold() in {"no", "not", "never", "unverified", "failed", "may", "might", "superseded", "replacing"}
               or word.casefold().endswith("n't") for word in omitted):
            continue
        return True
    return False


def score_response(case: dict[str, Any], trace: dict[str, Any], included: list[str], rules: dict[str, Any]) -> dict[str, Any]:
    if (not isinstance(trace, dict) or not isinstance(trace.get("answer"), str) or not isinstance(trace.get("claims"), list)
            or not isinstance(trace.get("actions"), list) or not all(isinstance(x, str) for x in trace["actions"])):
        raise ValueError("invalid_response_trace")
    sources = {s["id"]: s["content"] for s in case["sources"]}
    if len(included) != len(set(included)) or not set(included) <= sources.keys():
        raise ValueError("invalid_included_source_manifest")
    claim_checks = []
    for claim in trace["claims"]:
        if (not isinstance(claim, dict) or not isinstance(claim.get("text"), str)
                or not isinstance(claim.get("source_ids"), list)
                or not all(isinstance(x, str) for x in claim["source_ids"])):
            raise ValueError("invalid_historical_claim")
        cited = claim["source_ids"]
        available = bool(cited) and set(cited) <= set(included)
        matched = available and all(_quote_matches(claim["text"], sources[source_id]) for source_id in cited)
        claim_checks.append({"text": claim["text"], "source_ids": cited, "source_available": available,
                             "quote_supported": matched})
    answer = _normalize(trace["answer"])
    missing = [index for index, alternatives in enumerate(rules["required"])
               if not any(re.search(pattern, answer, re.I) for pattern in alternatives)]
    exception_spans = [match.span() for pattern in rules["negation_exceptions"] for match in re.finditer(pattern, answer, re.I)]
    # Exceptions cover only their explicit span. A later affirmative assertion
    # remains forbidden, even if an earlier sentence contains a valid denial.
    forbidden_matches = [match.group() for pattern in rules["forbidden"] for match in re.finditer(pattern, answer, re.I)
                         if not any(start <= match.start() and match.end() <= end for start, end in exception_spans)]
    forbidden_actions = sorted(set(trace["actions"]) & set(case["expected"]["forbidden_actions"]))
    unsupported = sum(not check["quote_supported"] for check in claim_checks)
    return {"scorer_version": SCORER_VERSION,
            "task_completion_proxy": not (missing or forbidden_matches or forbidden_actions or unsupported),
            "missing_requirements": missing, "forbidden_answer_matches": forbidden_matches,
            "unsupported_historical_claims": unsupported, "claim_checks": claim_checks,
            "forbidden_actions": forbidden_actions,
            "forbidden_context_sources": sorted(set(included) & set(case["expected"]["forbidden_sources"])),
            "semantic_review_required": True, "verified_task_completion": None}


def rescore(corpus: dict[str, Any], report: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != 1 or report.get("mode") != "live-response":
        raise ValueError("saved_live_response_report_required")
    if report.get("corpus_sha256") != content_hash(corpus):
        raise ValueError("saved_response_corpus_mismatch")
    cases = {case["id"]: case for case in corpus["cases"]}
    rows = report["cases"]
    if len(rows) != len(cases) or {row["id"] for row in rows} != set(cases):
        raise ValueError("saved_response_case_mismatch")
    scored = []
    for row in rows:
        if "response" not in row:
            raise ValueError("saved_response_required")
        scored.append({"id": row["id"], "original_answer_quality": row["answer_quality"],
                       "answer_quality": score_response(cases[row["id"]], row["response"], row["included_sources"], rubric["cases"][row["id"]])})
    return {"schema_version": 1, "scorer_version": SCORER_VERSION, "corpus_sha256": content_hash(corpus),
            "original_report_sha256": content_hash(report), "rubric_sha256": content_hash(rubric),
            "model_calls": 0, "cases": scored}


def main() -> None:
    from alphonse.agent_v2.evaluation.memory_replay import load_corpus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.corpus.resolve(), args.rubric.resolve()}:
        parser.error("rescore output must preserve inputs")
    corpus = load_corpus(args.corpus)
    rubric = load_rubric(args.rubric, corpus)
    result = rescore(corpus, json.loads(args.input.read_text()), rubric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
