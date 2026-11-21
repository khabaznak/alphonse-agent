import hashlib

from alphonse.agent_v2.evaluation.dojo_crm_database import CSV, LEADS, grade


def test_correct_rows_accept_any_column_order_and_extra_fields():
    rows = [[lead[2], lead[0], 'extra', lead[3], lead[1]] for lead in LEADS]
    fixture = {'status': 'observed', 'source_sha256': hashlib.sha256(CSV.encode()).hexdigest(),
               'database_status': 'observed', 'tables': {'any_name': {'rows': rows,
                   'columns': ['company', 'ID', 'notes', 'status', 'name'], 'update_probe': True}}, 'scan_complete': True}
    result = grade(fixture)
    assert result['effect_passed'] is True
    assert result['response_semantics'] == 'unassessed'
    assert grade({**fixture, 'source_sha256': 'wrong'})['effect_passed'] is False


def test_row_presence_alone_does_not_prove_schema_or_usability():
    rows = [list(lead) for lead in LEADS]
    fixture = {'status': 'observed', 'source_sha256': hashlib.sha256(CSV.encode()).hexdigest(),
               'database_status': 'observed', 'tables': {'archive': {'rows': rows,
                   'columns': ['id', 'name', 'company', 'status']}}, 'scan_complete': True}
    assert grade(fixture)['expected_leads_with_same_row_evidence'] == 4
    assert grade(fixture)['effect_passed'] is None
    assert grade({**fixture, 'tables': {'archive': {**fixture['tables']['archive'], 'update_probe': False}}})['effect_passed'] is None
    extra = {'rows': rows + [['fake', 'Fake', 'Imaginary', 'new']],
             'columns': ['id', 'name', 'company', 'status'], 'update_probe': True}
    assert grade({**fixture, 'tables': {'archive': extra}})['effect_passed'] is None


def test_unmatched_schema_is_not_automatically_a_failure():
    fixture = {'status': 'observed', 'source_sha256': hashlib.sha256(CSV.encode()).hexdigest(),
               'database_status': 'observed', 'tables': {'normalized': {'rows': [['P101']]}},
               'scan_complete': True}
    assert grade(fixture)['effect_passed'] is None
    assert grade({**fixture, 'database_status': 'missing'})['effect_passed'] is False
    assert grade({'status': 'unavailable'})['effect_passed'] is None
