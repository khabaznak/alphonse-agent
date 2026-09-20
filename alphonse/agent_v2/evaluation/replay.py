"""Fixture-isolated V2/V3 replay harness and comparison report writer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    goal: str
    required_capabilities: tuple[str, ...] = ()
    forbidden_capabilities: tuple[str, ...] = ()
    allowed_effects: tuple[str, ...] = ()
    forbidden_questions: tuple[str, ...] = ()
    forbidden_outcome: str = ""
    fixture: dict[str, Any] = field(default_factory=dict)
    expectations: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineTrace:
    """Observable result returned by an engine adapter; no hidden reasoning."""

    engine: str
    outcome: str = ""
    status: str = "completed"
    capabilities: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class EngineReplayRunner(Protocol):
    def __call__(
        self,
        case: EvaluationCase,
        fixture_root: Path,
        telemetry_sink: Callable[[dict[str, Any]], None],
    ) -> EngineTrace: ...


def load_corpus(path: str | Path) -> list[EvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported_evaluation_corpus_schema")
    cases: list[EvaluationCase] = []
    for raw in payload.get("cases") or []:
        known = {
            "id", "goal", "required_capabilities", "forbidden_capabilities",
            "allowed_effects", "forbidden_questions", "forbidden_outcome", "fixture",
        }
        cases.append(EvaluationCase(
            case_id=str(raw["id"]),
            goal=str(raw["goal"]),
            required_capabilities=tuple(map(str, raw.get("required_capabilities") or ())),
            forbidden_capabilities=tuple(map(str, raw.get("forbidden_capabilities") or ())),
            allowed_effects=tuple(map(str, raw.get("allowed_effects") or ())),
            forbidden_questions=tuple(map(str, raw.get("forbidden_questions") or ())),
            forbidden_outcome=str(raw.get("forbidden_outcome") or ""),
            fixture=dict(raw.get("fixture") or {}),
            expectations={key: value for key, value in raw.items() if key not in known},
        ))
    return cases


class ReplayHarness:
    """Run identical cases through two adapters using fresh temporary projects."""

    def __init__(self, *, v2_runner: EngineReplayRunner, v3_runner: EngineReplayRunner) -> None:
        self.runners = {"tactical_v2": v2_runner, "hierarchical_v3": v3_runner}

    def run(self, cases: list[EvaluationCase]) -> dict[str, Any]:
        rows = []
        for case in cases:
            runs = {engine: self._run_one(case, engine, runner) for engine, runner in self.runners.items()}
            rows.append({"case_id": case.case_id, "goal": case.goal, "runs": runs})
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": _summarize(rows),
            "cases": rows,
        }

    def write_reports(self, cases: list[EvaluationCase], output_dir: str | Path) -> tuple[Path, Path]:
        report = self.run(cases)
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "v3-comparison.json"
        markdown_path = target / "v3-comparison.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        return json_path, markdown_path

    @staticmethod
    def _run_one(case: EvaluationCase, engine: str, runner: EngineReplayRunner) -> dict[str, Any]:
        captured: list[dict[str, Any]] = []
        started = monotonic()
        with tempfile.TemporaryDirectory(prefix=f"alphonse-replay-{case.case_id}-{engine}-") as temp:
            fixture_root = Path(temp).resolve()
            _materialize_fixture(fixture_root, case.fixture)
            before_files = _snapshot_files(fixture_root / "project")
            try:
                trace = runner(case, fixture_root, lambda event: captured.append(dict(event)))
                if trace.engine != engine:
                    raise ValueError(f"runner_engine_mismatch:{trace.engine}")
            except Exception as exc:
                trace = EngineTrace(engine=engine, status="failed", error=f"{type(exc).__name__}: {exc}")
            after_files = _snapshot_files(fixture_root / "project")
            observed_effects = sorted(
                path for path in set(before_files) | set(after_files)
                if before_files.get(path) != after_files.get(path)
            )
            trace.effects = list(dict.fromkeys([*trace.effects, *observed_effects]))
        if captured:
            trace.telemetry.extend(captured)
        elapsed_ms = max(0, round((monotonic() - started) * 1000))
        violations = _validate_trace(case, trace)
        return {
            **asdict(trace),
            "elapsed_ms": elapsed_ms,
            "metrics": _metrics(trace.telemetry, elapsed_ms),
            "violations": violations,
            "passed": not violations and trace.status not in {"failed", "error"} and not trace.error,
        }


def _materialize_fixture(root: Path, fixture: dict[str, Any]) -> None:
    (root / "project").mkdir(parents=True, exist_ok=True)
    for relative, content in dict(fixture.get("files") or {}).items():
        path = _fixture_path(root / "project", str(relative))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
    attachments = root / "attachments"
    attachments.mkdir(exist_ok=True)
    for item in fixture.get("attachments") or []:
        asset_id = str(item.get("asset_id") or "asset")
        path = _fixture_path(attachments, asset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _fixture_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise ValueError(f"fixture_path_outside_root:{relative}")
    return candidate


def _snapshot_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def _validate_trace(case: EvaluationCase, trace: EngineTrace) -> list[str]:
    violations: list[str] = []
    capabilities = set(trace.capabilities)
    for capability in case.required_capabilities:
        if capability not in capabilities:
            violations.append(f"required_capability_missing:{capability}")
    for capability in case.forbidden_capabilities:
        if capability in capabilities:
            violations.append(f"forbidden_capability_used:{capability}")
    allowed = set(case.allowed_effects)
    for effect in trace.effects:
        path = PurePosixPath(str(effect).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            violations.append(f"effect_outside_fixture:{effect}")
        elif effect not in allowed:
            violations.append(f"effect_not_allowed:{effect}")
    questions = "\n".join(trace.questions).casefold()
    for forbidden in case.forbidden_questions:
        if forbidden.casefold() in questions:
            violations.append(f"forbidden_question:{forbidden}")
    if case.forbidden_outcome and case.forbidden_outcome.casefold() in trace.outcome.casefold():
        violations.append(f"forbidden_outcome:{case.forbidden_outcome}")
    expected_phase_shape = case.expectations.get("expected_phase_shape")
    if expected_phase_shape is not None and trace.metadata.get("phase_shape") != expected_phase_shape:
        violations.append("expected_phase_shape_mismatch")
    expected_metadata = {
        key.removeprefix("expected_"): value
        for key, value in case.expectations.items()
        if key.startswith("expected_") and key != "expected_phase_shape"
    }
    for key, expected in expected_metadata.items():
        if trace.metadata.get(key) != expected:
            violations.append(f"expected_{key}_mismatch")
    return violations


def _metrics(events: list[dict[str, Any]], elapsed_ms: int) -> dict[str, int]:
    inference = [event for event in events if event.get("event_type") == "inference"]
    tools = [event for event in events if event.get("event_type") == "tool"]
    provider_input = provider_output = 0
    for event in inference:
        usage = event.get("provider_usage") if isinstance(event.get("provider_usage"), dict) else {}
        provider_input += _first_int(usage, "input_tokens", "prompt_tokens")
        provider_output += _first_int(usage, "output_tokens", "completion_tokens")
    return {
        "elapsed_ms": elapsed_ms,
        "inference_calls": len(inference),
        "tool_calls": len(tools),
        "failed_inference_calls": sum(event.get("status") != "success" for event in inference),
        "failed_tool_calls": sum(event.get("status") != "success" for event in tools),
        "input_tokens_estimate": sum(int(event.get("input_tokens_estimate") or 0) for event in inference),
        "output_tokens_estimate": sum(int(event.get("output_tokens_estimate") or 0) for event in inference),
        "tool_schema_tokens_estimate": sum(int(event.get("tool_schema_tokens_estimate") or 0) for event in inference),
        "provider_input_tokens": provider_input,
        "provider_output_tokens": provider_output,
        "inference_duration_ms": sum(int(event.get("duration_ms") or 0) for event in inference),
        "tool_duration_ms": sum(int(event.get("duration_ms") or 0) for event in tools),
    }


def _first_int(value: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key not in value or value[key] is None:
            continue
        try:
            return int(value[key])
        except (TypeError, ValueError):
            continue
    return 0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"case_count": len(rows), "engines": {}}
    for engine in ("tactical_v2", "hierarchical_v3"):
        runs = [row["runs"][engine] for row in rows]
        summary["engines"][engine] = {
            "passed": sum(run["passed"] for run in runs),
            "failed": sum(not run["passed"] for run in runs),
            "inference_calls": sum(run["metrics"]["inference_calls"] for run in runs),
            "tool_calls": sum(run["metrics"]["tool_calls"] for run in runs),
            "estimated_tokens": sum(
                run["metrics"]["input_tokens_estimate"]
                + run["metrics"]["output_tokens_estimate"]
                + run["metrics"]["tool_schema_tokens_estimate"]
                for run in runs
            ),
            "elapsed_ms": sum(run["metrics"]["elapsed_ms"] for run in runs),
        }
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# V2 / V3 replay comparison", "", f"Generated: {report['generated_at']}", ""]
    lines += ["| Engine | Passed | Failed | Inference calls | Tool calls | Estimated tokens | Elapsed ms |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for engine, values in report["summary"]["engines"].items():
        lines.append(
            f"| {engine} | {values['passed']} | {values['failed']} | {values['inference_calls']} | "
            f"{values['tool_calls']} | {values['estimated_tokens']} | {values['elapsed_ms']} |"
        )
    lines += ["", "## Cases", ""]
    for row in report["cases"]:
        lines.append(f"### {row['case_id']}")
        lines.append("")
        for engine, run in row["runs"].items():
            result = "PASS" if run["passed"] else "FAIL"
            detail = ", ".join(run["violations"]) or run.get("error") or "no policy violations"
            lines.append(f"- {engine}: {result} — {detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_runner(reference: str) -> EngineReplayRunner:
    module_name, separator, attribute = reference.partition(":")
    if not separator:
        raise ValueError("runner reference must use module:attribute")
    runner = getattr(importlib.import_module(module_name), attribute)
    return runner() if isinstance(runner, type) else runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Alphonse V2 and V3 using isolated replay fixtures.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--v2-runner", required=True, help="Python module:callable replay adapter")
    parser.add_argument("--v3-runner", required=True, help="Python module:callable replay adapter")
    args = parser.parse_args(argv)
    harness = ReplayHarness(v2_runner=_load_runner(args.v2_runner), v3_runner=_load_runner(args.v3_runner))
    json_path, markdown_path = harness.write_reports(load_corpus(args.corpus), args.output_dir)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
