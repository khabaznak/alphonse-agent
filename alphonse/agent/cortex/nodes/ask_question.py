from __future__ import annotations

from typing import Any, Callable


def ask_question_node(
    state: dict[str, Any],
    *,
    run_ask_question_step: Callable[[dict[str, Any], dict[str, Any], dict[str, Any] | None, int | None], dict[str, Any]],
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None],
) -> dict[str, Any]:
    """Emit question for askQuestion step and park on pending interaction."""
    loop_state = state.get("ability_state")
    if not isinstance(loop_state, dict):
        return {}
    steps = loop_state.get("steps")
    if not isinstance(steps, list) or not steps:
        return {}
    idx_raw = state.get("selected_step_index")
    idx = idx_raw if isinstance(idx_raw, int) else next_step_index(steps, {"ready"})
    if idx is None or idx < 0 or idx >= len(steps):
        return {}
    step = steps[idx]
    if not isinstance(step, dict):
        return {}
    if str(step.get("tool") or "").strip() != "askQuestion":
        return {}
    return run_ask_question_step(state, step, loop_state, idx)


def ask_question_node_stateful(
    state: dict[str, Any],
    *,
    run_ask_question_step: Callable[[dict[str, Any], dict[str, Any], dict[str, Any] | None, int | None], dict[str, Any]],
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None],
) -> dict[str, Any]:
    return ask_question_node(
        state,
        run_ask_question_step=run_ask_question_step,
        next_step_index=next_step_index,
    )


def bind_answer_to_steps(
    steps: list[dict[str, Any]],
    bind: dict[str, Any],
    pending_key: str,
    answer: str,
) -> None:
    bound = False
    step_index = bind.get("step_index")
    param = str(bind.get("param") or pending_key or "answer").strip()
    if isinstance(step_index, int) and 0 <= step_index < len(steps):
        target = steps[step_index]
        params = target.get("parameters") if isinstance(target.get("parameters"), dict) else {}
        params[param] = answer
        target["parameters"] = params
        if target.get("status") == "incomplete" and not _has_missing_params(params):
            target["status"] = "ready"
        bound = True
    if bound:
        return
    for step in steps:
        if str(step.get("status") or "").lower() not in {"incomplete", "ready"}:
            continue
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        if param in params and (params[param] is None or (isinstance(params[param], str) and not params[param].strip())):
            params[param] = answer
            step["parameters"] = params
            if step.get("status") == "incomplete" and not _has_missing_params(params):
                step["status"] = "ready"
            return
    for step in steps:
        if str(step.get("status") or "").lower() != "incomplete":
            continue
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        for key, value in params.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                params[key] = answer
                step["parameters"] = params
                if not _has_missing_params(params):
                    step["status"] = "ready"
                return


def _has_missing_params(params: dict[str, Any]) -> bool:
    for value in params.values():
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
    return False
