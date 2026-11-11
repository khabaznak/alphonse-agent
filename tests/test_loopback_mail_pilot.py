"""Synthetic mail fixture checks; never connects to Gmail."""
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from alphonse.agent_v2.evaluation.loopback_mail_pilot import LoopbackMailPilot, evaluate


@pytest.mark.parametrize(('flags', 'count'), [((True, False, True, True), 3), ((False,), 0), ((), 0)])
def test_independent_answer_requires_observed_synthetic_source(flags, count):
    with LoopbackMailPilot(flags) as fixture:
        with urlopen(fixture.url, timeout=2) as response:
            assert json.load(response) == {'folder': 'INBOX', 'unread_count': count}
    result = evaluate(fixture, response={'unread_count': count}, expected_unread_count=count)
    assert result['passed'] and result['delivery'] == 'unassessed'
    assert not result['decision_qualified']


def test_guessed_answer_wrong_answer_wrong_key_and_bool_fail():
    with LoopbackMailPilot((True,)) as unobserved:
        pass
    assert not evaluate(unobserved, response={'unread_count': 1}, expected_unread_count=1)['passed']
    with LoopbackMailPilot((True,)) as observed:
        with urlopen(observed.url, timeout=2) as response:
            json.load(response)
    assert not evaluate(observed, response={'unread_count': 2}, expected_unread_count=1)['passed']
    assert not evaluate(observed, response={'unread_count': True}, expected_unread_count=1)['passed']
    assert not evaluate(observed, response={'unread_count': 1}, expected_unread_count=2)['passed']


def test_denied_paths_and_lifecycle():
    pilot = LoopbackMailPilot((True, False))
    with pilot:
        with pytest.raises(HTTPError):
            urlopen(pilot.url + 'other', timeout=2)
        with pytest.raises(RuntimeError):
            evaluate(pilot, response={'unread_count': 1}, expected_unread_count=1)
    assert pilot.audit() == [{'event': 'denied_request'}]
    with pytest.raises(RuntimeError):
        pilot.__enter__()


def test_bad_fixture_inputs_are_rejected():
    with pytest.raises(ValueError):
        LoopbackMailPilot((1,))
