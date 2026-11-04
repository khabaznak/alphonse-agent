"""Atomic mission judgments, independent of planner-authored acceptance criteria."""


def _noul(question, yes, no):
    return {'type': 'noul', 'instructions': question, 'criteria': {'true': yes, 'false': no}}


OUTCOME_QUESTIONS = {
    'mission_outcome_is_fulfilled': _noul(
        'Does recorded evidence fulfill the original goal and authenticated current-task steering?',
        'The requested result is verified. For advice-only, plan-only or capability-only requests, the prepared answer supplies the requested assessment without activating unrequested work.',
        'Delegated work remains undone. Manual instructions, proposals, attempts, dependency readiness and honest inability reports cannot replace the delegated result, even when generated acceptance criteria only require that report.'),
    'mission_requires_execution': _noul(
        'Does the user delegate an operation or a lookup for Alphonse to perform?',
        'The user requests execution, including read-only retrieval, a persisted change or recurring setup.',
        'The user requests only advice, a plan, a draft, an explanation or a capability assessment; the underlying operation is not delegated.'),
    'mission_is_capability_only': _noul(
        'Is the requested deliverable only an assessment of capability and prerequisites?',
        'The user asks whether or how Alphonse could perform an operation, without requesting activation.',
        'The user delegates work or requests a different deliverable.'),
    'capability_claims_are_supported': _noul(
        'Are capability claims in the prepared response supported by the recorded evidence?',
        'Claims are scoped to observed checks, available capabilities and explicit constraints, with untested alternatives identified as unverified; or there are no capability claims.',
        'An inability claim cites only itself, generated criteria, a restricted phase, tool curation, a missing dedicated tool or one failed route as proof of global inability. Evidence contradicts a capability claim.'),
    'lookup_claims_are_supported': _noul(
        'Does the evidence support the coverage claimed for lookup results in the prepared response?',
        'Results match successful retrieval. No-results claims have successful coverage of the requested scope. Partial coverage and uncertainty are explicitly scoped; or no lookup results are claimed.',
        'Failed lookup, unattempted work, omitted observations or partial coverage are described as verified absence. Results contradict observations.'),
    'mission_requires_recurring_setup': _noul(
        'Does the user request activation of recurring scheduled work?',
        'The user delegates creation or activation of recurring work.',
        'The user only requests a proposal, instructions or capability assessment, or work without recurring activation.'),
    'work_is_useful_progress': _noul(
        'Has recorded work produced useful intermediate progress toward the requested outcome?',
        'New relevant findings or verified intermediate effects advance the mission.',
        'Only acknowledgments, repeated plans or unsupported claims are present.'),
    'blocker_is_supported': _noul(
        'Does recorded evidence establish a blocker to the delegated outcome?',
        'A concrete failed prerequisite, access denial, unavailable resource or essential missing input prevents completion within the authorized scope.',
        'A blocker is asserted without supporting observations, contradicted by evidence or inferred only from curation or the current plan.'),
}
