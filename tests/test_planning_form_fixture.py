from copy import deepcopy

import pytest

from alphonse.agent_v2.evaluation.planning_form_fixture import FixtureDenied, FormFixture


def test_multifield_completion_requires_correct_values_and_readback():
    fixture = FormFixture()
    form = fixture.inspect('fixture-owner')
    fixture.fill('fixture-owner', form['version'], 'name', 'Synthetic Robin')
    with pytest.raises(FixtureDenied, match='incomplete_or_wrong_values'):
        fixture.submit('fixture-owner', form['version'])
    fixture.fill('fixture-owner', form['version'], 'code', 'fixture-17')
    fixture.submit('fixture-owner', form['version'])
    assert not fixture.oracle()['verified']
    assert fixture.verify('fixture-owner')['submitted'] == fixture.fields
    assert fixture.oracle() == {'verified': True, 'submission_count': 1, 'expected_values': True}
    with pytest.raises(FixtureDenied, match='effect_requires_reconciliation'):
        fixture.submit('fixture-owner', form['version'])


def test_changed_forms_denial_cancellation_and_restart_do_not_bypass_checks():
    fixture = FormFixture()
    version = fixture.inspect('fixture-owner')['version']
    fixture.fill('fixture-owner', version, 'name', 'Synthetic Robin')
    fixture.change_form({'name': 'Synthetic Robin', 'reference': 'fixture-17'})
    restored = FormFixture.restore(fixture.snapshot())
    with pytest.raises(FixtureDenied, match='stale_form'):
        restored.fill('fixture-owner', version, 'code', 'fixture-17')
    with pytest.raises(FixtureDenied, match='access_denied'):
        restored.inspect('other')
    restored.cancelled = True
    with pytest.raises(FixtureDenied, match='cancelled'):
        restored.fill('fixture-owner', restored.version, 'name', 'Synthetic Robin')
    assert restored.oracle()['submission_count'] == 0


def test_uncertain_external_effect_is_not_replayed_after_restart():
    fixture = FormFixture()
    version = fixture.inspect('fixture-owner')['version']
    for key, value in fixture.fields.items():
        fixture.fill('fixture-owner', version, key, value)
    assert fixture.submit('fixture-owner', version, ambiguous=True)['status'] == 'uncertain'
    restored = FormFixture.restore(deepcopy(fixture.snapshot()))
    with pytest.raises(FixtureDenied, match='effect_requires_reconciliation'):
        restored.submit('fixture-owner', version)
    assert restored.verify('fixture-owner')['submitted'] == restored.fields
    assert restored.oracle()['verified']
    assert restored.oracle()['submission_count'] == 1


def test_readback_before_submission_does_not_verify_a_later_effect():
    fixture = FormFixture()
    fixture.verify('fixture-owner')
    for key, value in fixture.fields.items():
        fixture.fill('fixture-owner', fixture.version, key, value)
    fixture.submit('fixture-owner', fixture.version)
    assert not fixture.oracle()['verified']
