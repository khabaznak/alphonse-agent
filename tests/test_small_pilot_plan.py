"""Small pilot agenda is planning information, not benchmark evidence."""
from copy import deepcopy
from collections import Counter

import pytest

from alphonse.agent_v2.evaluation.small_pilot_plan import (
    CASES, create_agenda, validate_agenda, traffic_fixture_smoke, readiness_receipt,
)


def test_paired_agenda_includes_all_arms_and_external_anchors():
    agenda = create_agenda(b'pilot protocol', seed='fixed')
    assert agenda == create_agenda(b'pilot protocol', seed='fixed')
    assert len(agenda['schedule']) == 36
    counts = Counter((row['case'], row['arm']) for row in agenda['schedule'])
    for case in CASES:
        for arm in ('v3', 'targeted_v3', 'single_loop'):
            assert counts[(case, arm)] == 2
    assert {case for case, arm in counts if arm == 'external'} == {'clock', 'traffic', 'prescription'}
    assert all(config['availability'] == 'unavailable' for config in agenda['arms'].values())
    assert agenda['kind'] == 'development_agenda_not_frozen_experiment'
    validate_agenda(agenda, b'pilot protocol')
    receipt = readiness_receipt(agenda, b'pilot protocol')
    assert len(receipt['slots']) == 36 and receipt['attempted'] == 0
    assert all(row['passed'] is None and row['attempt_state'] == 'not_started'
               for row in receipt['slots'])
    assert receipt['qualification'] == 'unassessed'


def test_changed_schedule_or_protocol_is_rejected():
    agenda = create_agenda(b'a', seed='fixed')
    changed = deepcopy(agenda)
    changed['schedule'][0]['arm'] = 'fabricated'
    with pytest.raises(ValueError, match='agenda_or_protocol_changed'):
        validate_agenda(changed, b'a')
    with pytest.raises(ValueError, match='agenda_or_protocol_changed'):
        validate_agenda(agenda, b'b')


def test_traffic_is_explicitly_scripted_wiring_smoke():
    result = traffic_fixture_smoke()
    assert result['pilot_executed'] is False
    assert result['decision_qualified'] is False
    assert result['result']['experiment_kind'] == 'offline_scripted_traffic_processor_development'
    assert all(row['availability'] == 'available' for row in result['result']['arms'].values())
