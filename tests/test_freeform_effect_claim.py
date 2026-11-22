"""Transport and fail-closed policy tests, not live semantic qualification."""
import pytest
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


@pytest.mark.parametrize('request_text,reply,probability,allowed', [
    ('Plan a hypothetical list in this chat.', 'Here is the revised list: apples.', 0.02, True),
    ('Actualiza el archivo guardado.', 'Ya actualicé el archivo.', 0.98, False),
    ('Modifie la liste enregistrée.', "J’ai mis la liste à jour.", 0.97, False),
    ('Plan a draft.', 'The list is updated.', 0.5, False),
    ('Explain the status.', 'Nothing was changed.', 0.02, True),
    ('Update the record.', 'Done.', float('nan'), False),
])
def test_effect_claim_probability_policy_and_context(request_text, reply, probability, allowed):
    captured = {}
    def transport(url, key, payload, timeout):
        captured.update(payload)
        return {'answers': {'external_effect_claim': {'type': 'noul', 'noul': probability}}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport)
    assert provider.review_freeform_effect_claim(request_context=request_text, response_text=reply) is allowed
    assert captured['state'] == {'request_context': request_text, 'response': reply}
    assert captured['questions']['external_effect_claim']['type'] == 'noul'


def test_effect_claim_unavailable_is_not_an_answer():
    attempts = []
    def transport(url, key, payload, timeout):
        attempts.append(timeout)
        raise RuntimeError('system_one_connection_failed')
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport)
    with pytest.raises(RuntimeError):
        provider.review_freeform_effect_claim(request_context='Change the list', response_text='Done')
    assert attempts == [3.0]
