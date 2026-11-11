"""Compare three scripted processor paths on public prescription fixtures.

This is evaluation plumbing, not an autonomous agent or clinical benchmark.
Both V3 arms and candidate receive a preselected scripted answer; no output
from this module is evidence that an agent discovered or reasoned about it.
"""
from __future__ import annotations

from .offline_three_arm import compare_three
from .prescription_candidate_adapter import PrescriptionCandidateAdapter
from .prescription_v3_adapter import PrescriptionV3Adapter
from .simplified_loop import AdapterUnavailable


class _CandidateScripts:
    offline = True
    revision = 'working-tree:single-loop:prescription-scripts-v1'

    def __init__(self, answers):
        self.answers = dict(answers)

    def __call__(self, case, root, emit):
        if case.get('id') not in self.answers:
            raise AdapterUnavailable('prescription_script_not_configured')
        decisions = iter([
            *({'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'path': p}}
              for p in ('household/index.md', 'household/prescription.md', 'household/treatment-start.md')),
            {'kind': 'answer', 'text': self.answers[case['id']]},
        ])
        return PrescriptionCandidateAdapter(lambda _: next(decisions))(case, root, emit)


def compare_prescriptions(corpus_path, *, scripted_answers):
    """The supplied scripts are shared across arms; they are not predictions."""
    arms = {
        'current_v3': PrescriptionV3Adapter(projection_enabled=False, scripted_answers=scripted_answers),
        'targeted_v3_projection': PrescriptionV3Adapter(projection_enabled=True, scripted_answers=scripted_answers),
        'single_loop': _CandidateScripts(scripted_answers),
    }
    report = compare_three(corpus_path, arms=arms)
    report['experiment_kind'] = 'offline_scripted_prescription_three_arm_plumbing'
    report['limitations'].append('All three arms use preselected answers, and the oracle does not establish natural-language quality, dosage, user delivery or clinical safety.')
    return report
