import json

import pytest

from alphonse.agent_v2.evaluation.browser_handoff_comparison import report, run_case, sequence_report


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
    assert (control["composition_calls"], sequence["composition_calls"]) == (7, 3)
    assert (control["recovery_guidance_calls"], sequence["recovery_guidance_calls"]) == (4, 0)
    assert control["observations"] == sequence["observations"] == 3
    changed = run_case(control=False, changed=True, sequence=True)
    assert changed["scenario_expectation_met"] and changed["unresolved_effect"]


def test_sequence_comparison_retains_each_safety_outcome():
    for qualify in (False, True):
        value = sequence_report(repeats=1, qualify=qualify)
        assert len(value["runs"]) == 12 and value["safety_oracle_failures"] == 0
        assert all(row["scenario_expectation_met"] for row in value["runs"])
        assert "SYNTHETIC_NAME" not in json.dumps(value)
        assert "SYNTHETIC_NUMBER" not in json.dumps(value)


@pytest.mark.parametrize("sequence", [False, True])
def test_verified_field_phase_can_complete_without_claiming_task_completion(sequence):
    normal = run_case(control=False, sequence=sequence, qualify=True)
    assert normal["scenario_expectation_met"] and normal["effect_oracle_pass"]
    assert normal["phase_completion"] == "qualified" and normal["task_completion"] == "not_qualified"
    assert normal["observations"] == 4  # includes the independent fixture check
    changed = run_case(control=False, sequence=sequence, qualify=True, scenario="changed")
    assert changed["scenario_expectation_met"] and changed["phase_completion"] == "not_qualified"


@pytest.mark.parametrize("sequence", [False, True])
def test_browser_malformed_response_retry_is_bounded_and_cancellable(sequence):
    recovered = run_case(control=False, scenario="malformed_once", sequence=sequence)
    assert recovered["scenario_expectation_met"] and recovered["verified_fields"] == 2
    assert recovered["composition_calls"] == (4 if sequence else 8)
    exhausted = run_case(control=False, scenario="malformed_always", sequence=sequence)
    assert exhausted["scenario_expectation_met"] and exhausted["composition_calls"] == 2
    assert exhausted["tool_calls"] == 0
    cancelled = run_case(control=False, scenario="malformed_cancelled", sequence=sequence)
    assert cancelled["scenario_expectation_met"] and cancelled["composition_calls"] == 1
    assert cancelled["tool_calls"] == 0
