"""Bounded read-only retrieval recovery; synthesis never establishes success."""
from .contracts import PhaseStatus, SideEffectClass
from .revealing import Capability

COMMUNICATION = {Capability.USER_RESPONSE.value, Capability.USER_INTERACTION.value}
INSTRUCTIONS = (
    "A retrieval_resolution directive ends repeated retrieval. Use the cumulative recorded evidence, "
    "including source dates, provenance, contradictions and explicit coverage gaps. State what the records "
    "support and what they do not establish; do not invent missing facts or equate historical records with "
    "current instructions. A scheduled event is not proof it occurred. If an essential missing fact prevents "
    "a safe answer, ask only for that specific fact/source using user_interaction. Otherwise prepare a useful "
    "answer with explicit limitations using user_response. If an acceptance criterion requires clarification "
    "under the observed gaps, ask that specific question; qualified prose alone does not satisfy it. "
    "Include useful supported findings and their limitations in the question_context.text when asking. "
    "Preserve the acceptance contract; this handoff "
    "does not verify the answer or fulfill a requested effect. Never claim a source was checked if it was not. "
    "For route remaining_source, you may plan at most one read_only retrieval subgoal for an unattempted "
    "source listed in the directive, followed by one communication subgoal; do not restart the stalled search. "
    "For route answer_or_clarify, plan exactly one communication subgoal with user_response effects only. "
    "No further retrieval or mutation is allowed in that resolution phase."
)


def prepare_handoff(task, state, outcome):
    """Offer one remaining-source pass and one synthesis pass, without changing criteria."""
    if outcome.status != PhaseStatus.BLOCKED or set(outcome.blockers) != {'tactical_stage_no_new_progress'}:
        return None
    allowed = {SideEffectClass.READ_ONLY, SideEffectClass.USER_RESPONSE}
    if any(set(stage.allowed_side_effects) - allowed for stage in state.phase.subgoals):
        return None
    reads = [stage for stage in state.phase.subgoals
             if set(stage.allowed_side_effects) == {SideEffectClass.READ_ONLY}
             and set(stage.allowed_capabilities) - COMMUNICATION]
    if not reads:
        return None
    rows = [row for row in state.evidence.entries if row.get('phase_id') == state.phase.phase_id
            and row.get('subgoal_id') in {stage.subgoal_id for stage in reads}]
    refs = [row['evidence_ref'] for row in rows if row.get('status') == 'success'
            and row.get('evidence_ref') and row.get('result') is not None]
    if not refs:
        return None
    history = task.metadata.get('v3_retrieval_handoffs') or []
    if any(row.get('route') == 'answer_or_clarify' or row.get('phase_id') == state.phase.phase_id for row in history):
        return None
    attempted = {row.get('subgoal_id') for row in rows}
    unattempted = [{'subgoal_id': stage.subgoal_id, 'objective': stage.objective,
                    'allowed_capabilities': list(stage.allowed_capabilities)}
                   for stage in reads if stage.subgoal_id not in attempted]
    route = 'remaining_source' if unattempted and not history else 'answer_or_clarify'
    receipt = {'phase_id': state.phase.phase_id, 'route': route, 'evidence_refs': refs,
               'unattempted_sources': unattempted, 'attempt': len(history) + 1}
    task.metadata['v3_retrieval_handoffs'] = [*history, receipt]
    return receipt


def validate_handoff(phase, directive):
    resolution = directive.get('retrieval_resolution') or {}
    if resolution.get('route') not in {'remaining_source', 'answer_or_clarify'}:
        return
    stages = phase.subgoals
    communication = stages[-1]
    if (not communication.allowed_capabilities or set(communication.allowed_capabilities) - COMMUNICATION
            or set(communication.allowed_side_effects) != {SideEffectClass.USER_RESPONSE}
            or phase.mutation_scope.allowed_paths or phase.mutation_scope.allow_external_effects):
        raise ValueError('retrieval_handoff_requires_answer_or_specific_question')
    if resolution['route'] == 'answer_or_clarify':
        valid = len(stages) == 1 and set(phase.authorized_capabilities) <= COMMUNICATION
    else:
        valid = len(stages) == 1 or (len(stages) == 2
            and set(stages[0].allowed_side_effects) == {SideEffectClass.READ_ONLY}
            and bool(set(stages[0].allowed_capabilities) - COMMUNICATION)
            and stages[0].subgoal_id in communication.depends_on)
    if not valid:
        raise ValueError('retrieval_handoff_requires_bounded_source_then_communication')
