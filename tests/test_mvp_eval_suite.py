from __future__ import annotations

import time
from dataclasses import dataclass

from alphonse.agent.cortex.graph import CortexGraph


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
        expected_route="direct_reply",
        tool_required=False,
    ),
    EvalScenario(
        message="Hi there!",
        expected_route="direct_reply",
        tool_required=False,
    ),
    EvalScenario(
        message="Set a reminder for me",
        expected_route="clarify",
        tool_required=False,
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
        expected_route="direct_reply",
        tool_required=False,
    ),
)


class _EvalLlm:
    def __init__(self) -> None:
        self.planning_calls_by_message: dict[str, int] = {}

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "routing controller for a conversational agent" in system_prompt:
            message = _extract_message(user_prompt, prefix="User message:")
            return self._first_decision_for_message(message)
        if "You are Alphonse planning engine." in system_prompt:
            message = _extract_message(user_prompt, prefix="MESSAGE:")
            self.planning_calls_by_message[message] = (
                self.planning_calls_by_message.get(message, 0) + 1
            )
            return self._plan_for_message(message)
        return "{}"

    @staticmethod
    def _first_decision_for_message(message: str) -> str:
        normalized = message.lower().strip()
        if normalized in {"can you speak spanish?", "hi there!", "thanks!"}:
            return (
                '{"route":"direct_reply","intent":"conversation.generic",'
                '"confidence":0.95,"reply_text":"Claro.","clarify_question":""}'
            )
        if normalized == "set a reminder for me":
            return (
                '{"route":"clarify","intent":"task.reminder.create",'
                '"confidence":0.82,"reply_text":"",'
                '"clarify_question":"When should I set it?"}'
            )
        if normalized in {"show your current runtime status.", "what can you do?"}:
            return (
                '{"route":"tool_plan","intent":"meta.query",'
                '"confidence":0.80,"reply_text":"",'
                '"clarify_question":""}'
            )
        return (
            '{"route":"direct_reply","intent":"conversation.generic",'
            '"confidence":0.60,"reply_text":"Okay.","clarify_question":""}'
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


def test_mvp_eval_suite() -> None:
    llm = _EvalLlm()
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
    assert metrics.unnecessary_tool_call_rate <= 0.10, _render_metrics(metrics)
    assert metrics.p95_latency_seconds <= 0.50, _render_metrics(metrics)
    assert sum(metrics.latency_buckets.values()) == metrics.total, _render_metrics(metrics)


def _classify_actual_route(*, result: dict[str, object], planning_calls: int) -> str:
    if planning_calls > 0:
        return "tool_plan"
    pending = result.get("pending_interaction")
    if isinstance(pending, dict):
        return "clarify"
    response_text = str(result.get("response_text") or "").strip()
    if response_text:
        return "direct_reply"
    return "tool_plan"


def _extract_message(prompt: str, *, prefix: str) -> str:
    marker = f"{prefix}\n"
    idx = prompt.find(marker)
    if idx < 0:
        return ""
    tail = prompt[idx + len(marker) :]
    return tail.split("\n\n", 1)[0].strip()


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
