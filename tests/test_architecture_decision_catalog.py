"""Structural integrity checks for public design material, not agent evaluation."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent / 'fixtures' / 'architecture_decision'


def test_public_design_catalog_is_hashed_and_labeled_unavailable():
    raw = (ROOT / 'development_catalog_v1.json').read_bytes()
    expected = (ROOT / 'development_catalog_v1.sha256').read_text().split()[0]
    assert hashlib.sha256(raw).hexdigest() == expected
    catalog = json.loads(raw)
    assert catalog['split'] == 'development'
    assert catalog['provenance']['held_out'] is False
    assert catalog['provenance']['sequestered'] is False
    assert len(catalog['cases']) >= 24
    assert len({case['id'] for case in catalog['cases']}) == len(catalog['cases'])
    assert all(case['execution_support'] == 'unavailable' and
               case['oracle_implementation'] == 'specification_only' for case in catalog['cases'])
    assert all({'correct_outcome', 'independent_observation', 'forbidden', 'deadline_seconds'} <=
               case['oracle_spec'].keys() for case in catalog['cases'])
    assert {'mail', 'traffic', 'prescription', 'mutation', 'lifecycle', 'communication'} <= {
        case['family'] for case in catalog['cases']}
