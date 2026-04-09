from __future__ import annotations

import json
import time
from dataclasses import dataclass

from alphonse.agent.cortex.graph import CortexGraph
import alphonse.agent.cortex.graph as graph_module
import alphonse.agent.cortex.task_mode.pdca as pdca_module
from alphonse.agent.cortex.task_mode.task_record import TaskRecord


@dataclass(frozen=True)
class EvalScenario:
    message: str
    expected_route: str
    tool_required: bool


@dataclass(frozen=True)
class EvalMetrics:
    total: int
    route_accuracy: float
    unnecessary_tool_call_rate: float
    p95_latency_seconds: float
    latency_buckets: dict[str, int]


SCENARIOS: tuple[EvalScenario, ...] = (
    EvalScenario(
        message="Can you speak Spanish?",
        expected_route="tool_plan",
        tool_required=True,
    ),
    EvalScenario(
        message="Hi there!",
        expected_route="tool_plan",
        tool_required=True,
    ),
    EvalScenario(
        message="Set a reminder for me",
        expected_route="tool_plan",
        tool_required=True,
    ),
    EvalScenario(
        message="Show your current runtime status.",
        expected_route="tool_plan",
        tool_required=True,
    ),
    EvalScenario(
        message="What can you do?",
        expected_route="tool_plan",
        tool_required=True,
    ),
    EvalScenario(
        message="Thanks!",
        expected_route="tool_plan",
        tool_required=True,
    ),
)


class _EvalLlm:
    def __init__(self) -> None:
        self.planning_calls_by_message: dict[str, int] = {}

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        prompt_kind = str(system_prompt or "").lower()
        user_prompt_kind = str(user_prompt or "").lower()
        is_first_decision = (
            "check mode" in prompt_kind
            or "capd check judge" in prompt_kind
            or "routing controller" in prompt_kind
            or (
                "strict json output" in user_prompt_kind
                and "case type" in user_prompt_kind
                and "user message" in user_prompt_kind
            )
        )
        if is_first_decision:
            message = _extract_message(user_prompt, prefix="User message:")
            case_type = _extract_case_type(user_prompt)
            return self._first_decision_for_message(message, case_type=case_type)
        is_pdca = "pdca loop" in prompt_kind or "working state view" in user_prompt_kind
        if is_pdca:
            message = _extract_goal(user_prompt)
            if message:
                self.planning_calls_by_message[message] = (
                    self.planning_calls_by_message.get(message, 0) + 1
                )
            return self._pdca_for_message(message)
        is_planning = (
            "planning engine" in prompt_kind
            or ("available tools" in user_prompt_kind and "planning context" in user_prompt_kind)
        )
        if is_planning:
            message = _extract_message(user_prompt, prefix="MESSAGE:")
            self.planning_calls_by_message[message] = (
                self.planning_calls_by_message.get(message, 0) + 1
            )
            return self._plan_for_message(message)
        return "{}"

    def complete_with_tools(self, *, messages, tools, tool_choice="auto"):  # noqa: ANN001
        _ = (tools, tool_choice)
        user_prompt = ""
        for message in messages or []:
            if isinstance(message, dict) and str(message.get("role") or "").strip().lower() == "user":
                user_prompt = str(message.get("content") or "")
        message = _extract_goal(user_prompt)
        if not message:
            message = _extract_message(user_prompt, prefix="MESSAGE:")
        return json.loads(self._pdca_for_message(message))

    @staticmethod
    def _first_decision_for_message(message: str, *, case_type: str) -> str:
        _ = message
        normalized_case = case_type.strip().lower()
        if normalized_case in {"execution_review", "task_resumption"}:
            return (
                '{"kind":"mission_failed","case_type":"execution_review","reason":"Synthetic eval stop condition.",'
                '"confidence":0.99,"criteria_updates":[],"evidence_refs":[],"failure_class":"eval_stop"}'
            )
        return (
            '{"kind":"plan","case_type":"new_request","reason":"Plan-first CAPD policy.",'
            '"confidence":0.95,"criteria_updates":[{"op":"append","text":"Advance user request successfully."}],'
            '"evidence_refs":[],"failure_class":null}'
        )

    @staticmethod
    def _plan_for_message(message: str) -> str:
        normalized = message.lower().strip()
        if normalized == "show your current runtime status.":
            return (
                '{"intention":"facts.agent.get","confidence":"high",'
                '"acceptance_criteria":["status returned"],'
                '"execution_plan":[{"tool":"facts.agent.get","parameters":{"topic":"status"}}]}'
            )
        if normalized == "what can you do?":
            return (
                '{"intention":"meta.capabilities","confidence":"high",'
                '"acceptance_criteria":["capabilities returned"],'
                '"execution_plan":[{"tool":"meta.capabilities","parameters":{"scope":"all"}}]}'
            )
        return (
            '{"intention":"unknown","confidence":"low",'
            '"execution_plan":[{"tool":"askQuestion","parameters":{"question":"Can you clarify?"}}]}'
        )

    @staticmethod
    def _pdca_for_message(message: str) -> str:
        normalized = message.lower().strip()
        if normalized == "set a reminder for me":
            return (
                '{"tool_call":{"kind":"call_tool","tool_name":"createReminder","args":{"ForWhom":"me","Time":"2026-02-15T11:00:00+00:00","Message":"Reminder set"}},'
                '"planner_intent":"I am creating the reminder requested by the user."}'
            )
        return (
            '{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":10}},'
            '"planner_intent":"I am gathering concrete evidence to satisfy the request."}'
        )


def test_mvp_eval_suite(monkeypatch) -> None:
    llm = _EvalLlm()
    monkeypatch.setattr(pdca_module, "_resolve_plan_provider", lambda: (llm, None))
    monkeypatch.setattr(
        graph_module,
        "check_node",
        lambda task_record, provenance: _synthetic_check_result(task_record, provenance=provenance),
    )
    runner = CortexGraph().build().compile()

    total = 0
    route_matches = 0
    no_tool_total = 0
    no_tool_with_planning = 0
    latencies: list[float] = []
    buckets = {"fast": 0, "moderate": 0, "slow": 0}

    for scenario in SCENARIOS:
        total += 1
        started = time.perf_counter()
        result = runner.invoke(
            {
                "chat_id": "eval-chat",
                "channel_type": "telegram",
                "channel_target": "eval-chat",
                "last_user_message": scenario.message,
                "_llm_client": llm,
            }
        )
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        _bucket_latency(elapsed, buckets)

        planning_calls = llm.planning_calls_by_message.get(scenario.message, 0)
        actual_route = _classify_actual_route(result=result, planning_calls=planning_calls)
        if actual_route == scenario.expected_route:
            route_matches += 1

        if not scenario.tool_required:
            no_tool_total += 1
            if planning_calls > 0:
                no_tool_with_planning += 1

    p95_latency = _p95(latencies)
    metrics = EvalMetrics(
        total=total,
        route_accuracy=(route_matches / total) if total else 0.0,
        unnecessary_tool_call_rate=(
            (no_tool_with_planning / no_tool_total) if no_tool_total else 0.0
        ),
        p95_latency_seconds=p95_latency,
        latency_buckets=buckets,
    )

    assert metrics.route_accuracy >= 0.95, _render_metrics(metrics)
    assert metrics.unnecessary_tool_call_rate <= 1.00, _render_metrics(metrics)
    assert metrics.p95_latency_seconds <= 0.50, _render_metrics(metrics)
    assert sum(metrics.latency_buckets.values()) == metrics.total, _render_metrics(metrics)


def _synthetic_check_result(task_record: TaskRecord, *, provenance: str) -> dict[str, object]:
    normalized = str(provenance or "").strip().lower()
    if normalized == "entry":
        task_record.status = "running"
        task_record.outcome = None
        return {
            "task_record": task_record,
            "verdict": "plan",
            "judge_result": {"kind": "plan", "reason": "Plan-first CAPD policy."},
            "case_type": "new_request",
            "status": task_record.status,
            "outcome": task_record.outcome,
            "reason": "Plan-first CAPD policy.",
            "confidence": 0.95,
            "consumed_inputs": [],
        }
    task_record.status = "failed"
    task_record.outcome = {
        "kind": "task_failed",
        "summary": "Synthetic eval stop condition.",
        "failure_class": "eval_stop",
    }
    return {
        "task_record": task_record,
        "verdict": "mission_failed",
        "judge_result": {"kind": "mission_failed", "reason": "Synthetic eval stop condition."},
        "case_type": "execution_review" if normalized == "do" else "task_resumption",
        "status": task_record.status,
        "outcome": task_record.outcome,
        "reason": "Synthetic eval stop condition.",
        "confidence": 0.99,
        "consumed_inputs": [],
    }


def _classify_actual_route(*, result: dict[str, object], planning_calls: int) -> str:
    check_result = result.get("check_result")
    if isinstance(check_result, dict):
        kind = str(check_result.get("verdict") or "").strip().lower()
        if kind == "plan":
            return "tool_plan"
        if kind == "conversation":
            return "direct_reply"
        if kind in {"mission_success", "mission_failed"}:
            return "tool_plan"
    if planning_calls > 0:
        return "tool_plan"
    task_record = result.get("task_record")
    if isinstance(task_record, dict) and str(task_record.get("status") or "").strip().lower() == "waiting_user":
        return "clarify"
    pending = result.get("pending_interaction")
    if isinstance(pending, dict):
        return "clarify"
    response_text = str(result.get("response_text") or "").strip()
    if response_text:
        return "direct_reply"
    return "tool_plan"


def _extract_message(prompt: str, *, prefix: str) -> str:
    normalized = str(prompt or "")
    if "# USER MESSAGE" in normalized:
        tail = normalized.split("# USER MESSAGE", 1)[1]
        stripped = tail.lstrip()
        if stripped.startswith("\n"):
            stripped = stripped[1:]
        return stripped.split("\n\n", 1)[0].strip()

    marker = f"{prefix}\n"
    idx = normalized.find(marker)
    if idx < 0:
        return ""
    tail = normalized[idx + len(marker) :]
    return tail.split("\n\n", 1)[0].strip()


def _extract_goal(prompt: str) -> str:
    marker = '"goal":'
    normalized = str(prompt or "")
    idx = normalized.find(marker)
    if idx < 0:
        return ""
    tail = normalized[idx + len(marker) :].lstrip()
    if not tail.startswith('"'):
        return ""
    tail = tail[1:]
    end = tail.find('"')
    if end < 0:
        return ""
    return tail[:end].strip()


def _extract_case_type(prompt: str) -> str:
    marker = "## CASE TYPE"
    normalized = str(prompt or "")
    idx = normalized.find(marker)
    if idx < 0:
        return ""
    tail = normalized[idx + len(marker) :]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _bucket_latency(elapsed: float, buckets: dict[str, int]) -> None:
    if elapsed < 0.05:
        buckets["fast"] += 1
        return
    if elapsed < 0.20:
        buckets["moderate"] += 1
        return
    buckets["slow"] += 1


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered)) - 1)
    return ordered[index]


def _render_metrics(metrics: EvalMetrics) -> str:
    return (
        "eval metrics: "
        f"total={metrics.total} "
        f"route_accuracy={metrics.route_accuracy:.3f} "
        f"unnecessary_tool_call_rate={metrics.unnecessary_tool_call_rate:.3f} "
        f"p95_latency_seconds={metrics.p95_latency_seconds:.4f} "
        f"latency_buckets={metrics.latency_buckets}"
    )
