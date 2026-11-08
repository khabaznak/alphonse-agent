import json

import pytest

from alphonse.agent_v2.evaluation.browser_handoff_comparison import report, run_case


def test_control_and_continuation_preserve_verified_partial_effects():
    control = run_case(control=True)
    continued = run_case(control=False)
    assert control["scenario_expectation_met"] and continued["scenario_expectation_met"]
    assert (control["scripted_phase_replans"], continued["scripted_phase_replans"]) == (3, 0)
    assert (control["observations"], continued["observations"]) == (3, 3)
    assert control["repeated_observations"] == continued["repeated_observations"] == 0
    assert control["verified_fields"] == continued["verified_fields"] == 2
    assert control["tool_calls"] == continued["tool_calls"] == 6
    assert control["task_completion"] == continued["task_completion"] == "not_qualified"


@pytest.mark.parametrize("control", [True, False])
def test_changed_form_remains_unresolved_without_dispatch(control):
    result = run_case(control=control, changed=True)
    assert result["scenario_expectation_met"] and result["unresolved_effect"]
    assert result["verified_fields"] == 0 and not result["effect_oracle_pass"]
    assert result["tool_calls"] == 1 and result["phase_outcomes"] == ["blocked"]


def test_report_is_bounded_and_does_not_expose_fixture_values():
    value = report(repeats=1)
    assert len(value["runs"]) == 4
    assert "SYNTHETIC_NAME" not in json.dumps(value) and "SYNTHETIC_NUMBER" not in json.dumps(value)
    for invalid in (0, 101, True):
        with pytest.raises(ValueError, match="invalid_repeats"):
            report(repeats=invalid)


def test_fill_sequence_comparison_preserves_verified_effects():
    control = run_case(control=False)
    sequence = run_case(control=False, sequence=True)
    assert control["effect_oracle_pass"] and sequence["effect_oracle_pass"]
    assert (control["scripted_compositions"], sequence["scripted_compositions"]) == (7, 3)
    assert control["observations"] == sequence["observations"] == 3
    changed = run_case(control=False, changed=True, sequence=True)
    assert changed["scenario_expectation_met"] and changed["unresolved_effect"]
