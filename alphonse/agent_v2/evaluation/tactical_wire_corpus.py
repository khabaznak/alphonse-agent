"""Frozen synthetic response corpus; this is not a real-browser or model benchmark."""
from __future__ import annotations

import json
from statistics import median
from time import perf_counter
from typing import Any

from alphonse.agent_v2.core.inference.openai_codex import _try_parse_json_object
from alphonse.agent_v2.core.inference.tactical_wire import decode
from alphonse.agent_v2.core.intelligence.v3.tactical_contract import valid_tactical_response

Q = {"question_id": "q", "type": "noul", "instructions": "Was the fixture observed?",
     "criteria": {"true": "Observed", "false": "Not observed"}}
A = {"tool_id": "native.browser_workflow_fill", "arguments": {"target": "fixture"}, "acceptance_questions": [Q]}
W = {"action": {"tool_id": A["tool_id"], "arguments_json": json.dumps(A["arguments"]), "acceptance_questions": [Q]},
     "actions": [], "browser_fill_sequence": [], "uncertainty_assessment": None}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


# Expected refers only to a syntactically safe, permitted proposal, not a completed task.
CORPUS = (
    ("valid", [_json(A)], [_json(W)], True),
    ("encoding_recovered", ['{', _json(A)], ['{', _json(W)], True),
    ("encoding_exhausted", ['{', '{'], ['{', '{'], False),
    ("non_object", ['[]'], ['[]', '[]'], False),
    ("nested_fragment", ['prefix ' + _json(A)], ['prefix ' + _json(W), 'prefix ' + _json(W)], False),
    ("missing_fields", [_json({'tool_id': A['tool_id']})], [_json({'action': {}}), _json({'action': {}})], False),
    ("bad_arguments", [_json({**A, 'arguments': {'wrong': 'fixture'}})],
        [_json({**W, 'action': {**W['action'], 'arguments_json': '{"wrong":"fixture"}'}})], False),
    ("unsafe_tool", [_json({**A, 'tool_id': 'native.unrevealed'})],
        [_json({**W, 'action': {**W['action'], 'tool_id': 'native.unrevealed'}})], False),
)


def _eligible(value: Any) -> bool:
    return (valid_tactical_response(value) and value.get('tool_id') == A['tool_id']
            and value.get('arguments') == A['arguments'])


def _run(outputs: list[str], structured: bool) -> tuple[bool, int]:
    calls = 0
    for output in outputs[:2]:
        calls += 1
        try:
            if structured:
                obj = json.loads(output)
                parsed = decode(obj)
            else:
                parsed = _try_parse_json_object(output)
                if parsed is None:
                    raise ValueError('encoding')
        except (ValueError, TypeError):
            continue
        return _eligible(parsed), calls
    return False, calls


def report(repeats: int = 20) -> dict[str, Any]:
    if type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError('invalid_repeats')
    arms: dict[str, Any] = {}
    for name, structured, index in [('legacy_parser', False, 1), ('structured_wire', True, 2)]:
        times, calls, wrong, unsafe = [], 0, 0, 0
        failures = []
        for case in CORPUS:
            for _ in range(repeats):
                start = perf_counter()
                accepted, count = _run(case[index], structured)
                times.append((perf_counter() - start) * 1000)
                calls += count
                if accepted != case[3]:
                    wrong += 1
                    unsafe += int(accepted and not case[3])
                    if case[0] not in failures:
                        failures.append(case[0])
        arms[name] = {'cases': len(CORPUS) * repeats, 'wrong': wrong, 'unsafe_accepts': unsafe,
                      'failure_cases': failures, 'scripted_provider_calls': calls,
                      'median_local_ms': round(median(times), 4)}
    return {'repeats': repeats, 'arms': arms,
            'note': 'Scripted responses; no tool execution, provider latency, task success, or live browser measured.'}


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
