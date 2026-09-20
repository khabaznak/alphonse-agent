from __future__ import annotations

import json
from pathlib import Path

from alphonse.agent_v2.evaluation import EngineTrace, EvaluationCase, ReplayHarness, load_corpus


def test_replay_harness_isolates_fixtures_and_writes_comparison_reports(tmp_path: Path) -> None:
    roots: list[Path] = []

    def runner(engine: str):
        def run(case, fixture_root, emit):
            roots.append(fixture_root)
            assert (fixture_root / "project" / "backlog.md").read_text() == "Solar: open"
            emit({
                "event_type": "inference", "status": "success", "input_tokens_estimate": 10,
                "output_tokens_estimate": 2, "tool_schema_tokens_estimate": 3, "duration_ms": 4,
                "provider_usage": {"prompt_tokens": 8, "completion_tokens": 2},
            })
            emit({"event_type": "tool", "status": "success", "duration_ms": 1})
            return EngineTrace(
                engine=engine,
                outcome="complete",
                capabilities=["project_record_search", "exact_text_mutation"],
                effects=["backlog.md"],
                metadata={"phase_shape": ["locate", "update", "verify"]},
            )
        return run

    case = EvaluationCase(
        case_id="solar", goal="Complete solar", fixture={"files": {"backlog.md": "Solar: open"}},
        required_capabilities=("project_record_search", "exact_text_mutation"),
        allowed_effects=("backlog.md",), expectations={"expected_phase_shape": ["locate", "update", "verify"]},
    )
    harness = ReplayHarness(v2_runner=runner("tactical_v2"), v3_runner=runner("hierarchical_v3"))

    json_path, markdown_path = harness.write_reports([case], tmp_path / "reports")
    report = json.loads(json_path.read_text())

    assert len(set(roots)) == 2
    assert all(not root.exists() for root in roots)
    assert report["summary"]["engines"]["tactical_v2"]["estimated_tokens"] == 15
    assert report["cases"][0]["runs"]["tactical_v2"]["metrics"]["provider_input_tokens"] == 8
    assert report["cases"][0]["runs"]["hierarchical_v3"]["passed"] is True
    assert "V2 / V3 replay comparison" in markdown_path.read_text()


def test_replay_harness_reports_forbidden_capability_effect_and_outcome() -> None:
    def bad_runner(engine: str):
        def run(case, fixture_root, emit):
            (fixture_root / "project" / "unreported.md").write_text("mutation")
            return EngineTrace(
                engine=engine,
                outcome="silent success",
                capabilities=["home_automation"],
                effects=["../outside.md"],
            )
        return run

    case = EvaluationCase(
        case_id="guardrails", goal="Safe work", forbidden_capabilities=("home_automation",),
        allowed_effects=(), forbidden_outcome="silent success",
    )
    report = ReplayHarness(
        v2_runner=bad_runner("tactical_v2"), v3_runner=bad_runner("hierarchical_v3"),
    ).run([case])
    violations = report["cases"][0]["runs"]["hierarchical_v3"]["violations"]

    assert "forbidden_capability_used:home_automation" in violations
    assert "effect_outside_fixture:../outside.md" in violations
    assert "effect_not_allowed:unreported.md" in violations
    assert "forbidden_outcome:silent success" in violations


def test_loads_the_checked_in_v3_corpus() -> None:
    cases = load_corpus(Path(__file__).parent / "fixtures" / "v3_evaluation_cases.json")
    assert len(cases) == 9
    assert cases[0].case_id == "solar-project-completion"
