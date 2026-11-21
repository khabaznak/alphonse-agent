import pytest

from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


@pytest.mark.parametrize('answer,accepted', [
    ({'type': 'noul', 'noul': 0.05}, True),
    ({'type': 'noul', 'noul': 0.5}, False),
    ({'type': 'noul', 'noul': 0.95}, False),
    ({'type': 'noul', 'noul': float('nan')}, False),
    ({'type': 'noul', 'noul': True}, False),
    ({'type': 'choice', 'noul': 0.05}, False),
    ({}, False),
])
def test_effect_claim_review_uses_probability_and_fails_closed(answer, accepted):
    observed = {}
    def transport(url, key, payload, timeout):
        observed.update(payload)
        return {'answers': {'completed_effect_claim': answer}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport)
    assert provider.review_clarification_effect_claim(
        message='Recuérdame llamar al dentista',
        clarification='¿Cuándo quieres que te recuerde llamar al dentista?',
    ) is accepted
    assert observed['state']['clarification'].startswith('¿Cuándo')
    assert observed['questions']['completed_effect_claim']['type'] == 'noul'


def test_unavailable_clarification_review_has_bounded_single_attempt():
    attempts = []
    def transport(url, key, payload, timeout):
        attempts.append(timeout)
        raise RuntimeError('system_one_connection_failed')
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport)
    with pytest.raises(RuntimeError, match='system_one_connection_failed'):
        provider.review_clarification_effect_claim(message='Please remind me', clarification='When?')
    assert attempts == [3.0]
