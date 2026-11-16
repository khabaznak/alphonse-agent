"""Keep the frozen synthetic corpus available after retiring V3 runners."""

from alphonse.agent_v2.evaluation.boundary_corpus import load_boundaries
from alphonse.agent_v2.evaluation.heldout_memory import load_heldout


def test_frozen_boundary_and_heldout_cases_remain_available():
    boundary = load_boundaries()
    heldout = load_heldout()
    assert boundary['synthetic'] is True
    assert heldout['synthetic'] is True
    assert len(boundary['cases']) == 41
    assert len(heldout['cases']) == 100
