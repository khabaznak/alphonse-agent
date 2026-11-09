import json

import pytest

from alphonse.agent_v2.evaluation.browser_metrics import summarize


def test_summary_allowlists_categories_and_never_copies_sensitive_values():
    secret = 'private-cookie-page-text-selector-password'
    events = [
        {"task_id": "t", "event_type": "inference", "purpose": "phase_planning", "duration_ms": 20,
         "status": "failed", "error": secret, "metadata": {"page": secret}},
        {"task_id": "t", "event_type": "tool", "tool_id": "native.browser_workflow_observe", "duration_ms": 5,
         "status": "success", "arguments": secret, "result": secret},
        {"task_id": "other", "event_type": "tool", "tool_id": "native.browser_workflow_fill", "duration_ms": 8, "status": "success"},
        {"task_id": "t", "event_type": "tool", "tool_id": "native.browser_workflow_" + secret, "duration_ms": 1, "status": "success"},
    ]
    result = summarize(events, task_id="t")
    assert result == {"schema_version": 1, "spans": {
        "browser.observe": {"count": 1, "failed": 0, "elapsed_ms": 5},
        "inference.phase_planning": {"count": 1, "failed": 1, "elapsed_ms": 20},
    }}
    assert secret not in json.dumps(result)
    assert '"t"' not in json.dumps(result)


def test_summary_rejects_malformed_events_and_requires_task_scope():
    with pytest.raises(ValueError):
        summarize([], task_id="")
    events = [{"task_id": "t", "event_type": "inference", "purpose": "tactical_action", "duration_ms": value, "status": "success"}
              for value in (True, -1, 86_400_001, float("nan"), "10")]
    events += [{"task_id": "t", "event_type": "inference", "purpose": "tactical_action", "duration_ms": 10, "status": "private"}]
    assert summarize(events, task_id="t")["spans"] == {}


def test_inclusive_orchestration_spans_do_not_include_exception_text():
    from types import SimpleNamespace
    from alphonse.agent_v2.core.intelligence.v3.measurement import measured_span

    events = []
    context = SimpleNamespace(emit_telemetry=events.append)
    task = SimpleNamespace(task_id="t")
    with pytest.raises(ValueError, match="private value"):
        with measured_span(context, task, "phase_planning"):
            raise ValueError("private value")
    assert events[0]["status"] == "raised"
    assert "private value" not in json.dumps(events)
    assert summarize(events, task_id="t")["spans"]["orchestration.phase_planning"]["failed"] == 1


def test_telemetry_sink_failure_does_not_break_execution_or_mask_error():
    from types import SimpleNamespace
    from alphonse.agent_v2.core.intelligence.v3.measurement import measured_span

    def broken(_event):
        raise RuntimeError("sink failure")
    context = SimpleNamespace(emit_telemetry=broken)
    task = SimpleNamespace(task_id="t")
    with measured_span(context, task, "tactical_composition"):
        pass
    with pytest.raises(ValueError, match="original"):
        with measured_span(context, task, "tactical_composition"):
            raise ValueError("original")
    with pytest.raises(ValueError, match="unknown_measurement_operation"):
        with measured_span(context, task, "secret"):
            pass


def test_disposable_probe_reports_blocked_effect_and_sanitized_measurements(tmp_path):
    from alphonse.agent_v2.evaluation.browser_probe import run_once

    result = run_once(tmp_path)
    assert result["outcome"] == "blocked"
    assert result["reason"] == "reconciliation_required"
    assert result["observations"] == result["fills"] == 1
    assert result["spans"]["browser.fill"]["count"] == 1
    assert result["spans"]["orchestration.tactical_composition"]["count"] == 2
    encoded = json.dumps(result)
    assert "synthetic name" not in encoded
    assert "1_1" not in encoded


def test_phase_planning_measurement_wraps_actual_processor_boundary(monkeypatch):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.intelligence.v3 import processor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState

    events = []
    task = TaskState(task_id="t", user="u")
    context = CoreLoopContext(messages=InMemoryMessageQueue(), telemetry_sink=events.append)
    def fail(_task, _context):
        raise ValueError("private planning content")
    monkeypatch.setattr(processor, "plan_phase", fail)
    with pytest.raises(ValueError):
        processor.HierarchicalCAPDProcessor._new_state(task, context)
    assert events == [{"event_type": "v3_measurement", "task_id": "t", "operation": "phase_planning",
                       "status": "raised", "duration_ms": events[0]["duration_ms"]}]
    assert "private planning content" not in json.dumps(events)


def test_field_probe_reports_verified_phase_not_task_completion():
    from alphonse.agent_v2.evaluation.browser_field_probe import run
    result = run(2)
    assert result["task_result"] == "not_verified"
    assert all(row["outcome"] == "verified_field_phase" and row["verified_fields"] == 2 for row in result["runs"])
    assert all(row["spans"]["browser.fill"]["count"] == 2 for row in result["runs"])
    assert "SYNTHETIC_NAME" not in json.dumps(result)
    assert "SYNTHETIC_NUMBER" not in json.dumps(result)
    for invalid in (0, 101, True):
        with pytest.raises(ValueError):
            run(invalid)


def test_field_probe_never_promotes_incomplete_oracle(monkeypatch):
    from alphonse.agent_v2.evaluation import browser_field_probe
    real_run_case = browser_field_probe.run_case
    def incomplete(**kwargs):
        result = real_run_case(**kwargs)
        result["effect_oracle_pass"] = False
        return result
    monkeypatch.setattr(browser_field_probe, "run_case", incomplete)
    result = browser_field_probe.run(1)
    assert result["runs"][0]["outcome"] == "unresolved"
    assert result["task_result"] == "not_verified"
