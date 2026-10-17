"""Explicit opt-in live responses to synthetic memory context; no tool execution.

Provider adapters are trusted Python and must make one inference with tools
 disabled. This module is deliberately separate from the offline entrypoint.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from alphonse.agent_v2.evaluation.memory_replay import load_corpus, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--responder", help="Explicit provider adapter module:function")
    args = parser.parse_args()
    if not args.live or not args.responder:
        parser.error("--live and --responder must be supplied together")
    corpus = load_corpus(args.corpus)
    module, name = args.responder.split(":", 1)
    responder = getattr(importlib.import_module(module), name)
    report = run(corpus, responder=responder, timing=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
