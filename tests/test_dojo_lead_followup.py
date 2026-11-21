import csv

import pytest

from alphonse.agent_v2.evaluation.dojo_lead_followup import goal, grade, readback, seed


def test_independent_scoring_checks_result_and_preserves_sources(tmp_path):
    workspace = seed(tmp_path)
    assert 'contact-history' in goal(workspace)
    with (workspace / 'follow-up.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerows([['name', 'company'], ['Devin Shaw', 'Juniper Works'],
                          ['Beatriz Cruz', 'Cedar Labs']])
    assert grade(readback(workspace))['effect_passed'] is True
    with (workspace / 'follow-up.csv').open('a', newline='') as stream:
        csv.writer(stream).writerow(['Asha Patel', 'Acorn Systems'])
    assert grade(readback(workspace))['checks']['correct_followup_list'] is False
    (workspace / 'leads.csv').write_text('modified')
    assert grade(readback(workspace))['checks']['source_files_unchanged'] is False


def test_symlinked_result_is_not_accepted_and_fixture_never_overwrites(tmp_path):
    workspace = seed(tmp_path)
    with pytest.raises(FileExistsError):
        seed(tmp_path)
    (workspace / 'follow-up.csv').symlink_to(workspace / 'leads.csv')
    assert grade(readback(workspace))['effect_passed'] is False


def test_missing_readback_is_not_a_pass():
    assert grade({'status': 'unavailable'})['effect_passed'] is None
