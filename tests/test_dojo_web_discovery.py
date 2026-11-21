import requests
import pytest

from alphonse.agent_v2.evaluation.dojo_web_discovery import scenario, synthetic_search_service
from alphonse.agent_v2.evaluation.owner_dojo_shopping import run


def test_synthetic_search_reports_requests_and_does_not_claim_semantic_success():
    with synthetic_search_service() as (base, calls):
        reply = requests.get(base + '/search', params={'q': 'temperature Virdale Beacon', 'format': 'json'}, timeout=3)
        assert reply.status_code == 200
        assert '28 °C' in reply.json()['results'][0]['content']
        case = scenario(calls)
        grade = case.grade(case.readback(None))
        assert grade['web_search_requests'] == 1
        assert grade['matching_search_requests'] == 1
        assert grade['effect_passed'] is None
        assert grade['response_semantics'] == 'requires_manual_review'


@pytest.mark.parametrize('url', ['https://example.com', 'http://localhost:1234',
                                 'http://127.0.0.1:1234/path', 'http://127.0.0.1'])
def test_fixture_rejects_non_loopback_or_ambiguous_search_settings(url):
    with pytest.raises(ValueError, match='dojo_synthetic_web_settings_invalid'):
        run(image_id='synthetic', docker_socket='/tmp/none', emit=lambda _: None,
            full_registry=True, synthetic_web_base_url=url)
