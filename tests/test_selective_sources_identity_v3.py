"""Privacy and interpretation of the public development-case diagnostic."""
import pytest

from alphonse.agent_v2.evaluation.selective_sources_identity_v3 import answer_shape


@pytest.mark.parametrize(('response', 'expected'), [
    ('{"status":"known","pharmacy":"synthetic-private-value"}',
     {'format': 'object', 'status': 'known', 'pharmacy': 'string'}),
    ('{"status":"unknown","pharmacy":null}',
     {'format': 'object', 'status': 'unknown', 'pharmacy': 'null'}),
    ('{"status":"synthetic-private-value","pharmacy":23}',
     {'format': 'object', 'status': 'other', 'pharmacy': 'other_type'}),
    ('synthetic-private-value', {'format': 'invalid_json'}),
    ('', {'format': 'no_answer'}),
])
def test_answer_shape_is_categorical_and_does_not_retain_content(response, expected):
    result = answer_shape(response)
    assert result == expected
    assert 'synthetic-private-value' not in repr(result)
