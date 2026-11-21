import hashlib

from alphonse.agent_v2.evaluation.dojo_corrected_plan import NOTES, grade


def test_fixture_integrity_does_not_claim_answer_success():
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in NOTES.items()}
    result = grade({'status': 'observed', 'note_sha256': hashes})
    assert result == {'effect_passed': None, 'fixture_integrity': 'unchanged', 'response_semantics': 'unassessed'}
    altered = dict(hashes, **{'visit-update.txt': 'modified'})
    assert grade({'status': 'observed', 'note_sha256': altered})['fixture_integrity'] == 'changed'
    assert grade({'status': 'unavailable'})['fixture_integrity'] == 'unavailable'
