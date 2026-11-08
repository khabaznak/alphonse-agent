"""Repeat verified synthetic field phases; this does not submit the form."""
from __future__ import annotations

import json
from statistics import median

from alphonse.agent_v2.evaluation.browser_handoff_comparison import run_case


def run(repeats: int = 5) -> dict:
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("invalid_repeats")
    rows = []
    for _ in range(repeats):
        result = run_case(control=False, sequence=True, qualify=True)
        # Export only static labels and numeric aggregates. Do not copy arbitrary
        # exception text, model responses, page snapshots, task IDs, or targets.
        verified = (result["phase_completion"] == "qualified" and result["effect_oracle_pass"]
                    and result["scenario_expectation_met"] and result["verified_fields"] == 2
                    and not result["unresolved_effect"])
        rows.append({"outcome": "verified_field_phase" if verified else "unresolved",
                     "elapsed_ms": round(result["elapsed_seconds"] * 1000, 3),
                     "compositions": result["composition_calls"], "tool_calls": result["tool_calls"],
                     "observations": result["observations"], "repeated_observations": result["repeated_observations"],
                     "verified_fields": result["verified_fields"], "spans": result["measurement_spans"]})
    return {"kind": "synthetic_field_phase_only", "runs": rows,
            "median_elapsed_ms": median(row["elapsed_ms"] for row in rows),
            "task_result": "not_verified"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
