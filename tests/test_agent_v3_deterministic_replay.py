from __future__ import annotations

from pathlib import Path

from alphonse.agent_v2.evaluation.deterministic_adapters import v2_runner, v3_runner
from alphonse.agent_v2.evaluation.replay import ReplayHarness, load_corpus


def test_checked_in_corpus_runs_through_real_v2_and_v3_processors() -> None:
    corpus = Path(__file__).parent / "fixtures" / "v3_evaluation_cases.json"

    report = ReplayHarness(v2_runner=v2_runner, v3_runner=v3_runner).run(load_corpus(corpus))

    assert report["summary"]["case_count"] == 9
    assert report["summary"]["engines"]["tactical_v2"]["failed"] == 0
    assert report["summary"]["engines"]["hierarchical_v3"]["failed"] == 0
    assert report["summary"]["engines"]["hierarchical_v3"]["tool_calls"] < report["summary"]["engines"]["tactical_v2"]["tool_calls"]
    assert all(
        not run["violations"]
        for case in report["cases"]
        for run in case["runs"].values()
    )
