import hashlib
from types import SimpleNamespace

from alphonse.agent_v2.evaluation.dojo_resourcefulness_followup import seed_scenario


def test_document_challenge_places_information_in_conversation_and_source(tmp_path):
    turns, prompt, fixture = seed_scenario('document-followup', tmp_path, None)
    assert fixture['context_marker'] in turns[0][1]
    assert fixture['context_marker'] not in prompt
    assert fixture['reading'] not in prompt
    assert fixture['reading'] in fixture['file'].read_text()
    assert hashlib.sha256(fixture['file'].read_bytes()).hexdigest() == fixture['digest']
    assert 'bash' not in prompt.lower()


def test_device_challenge_requires_using_prior_context(tmp_path):
    device = SimpleNamespace(server=SimpleNamespace(server_port=12345), reading='47')
    turns, prompt, fixture = seed_scenario('device-followup', tmp_path, device)
    assert fixture['context_marker'] in turns[0][1]
    assert fixture['context_marker'] not in prompt
    assert fixture['reading'] not in prompt
    assert 'bash' not in prompt.lower()
