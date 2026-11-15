"""V3 bridge exports, loaded only when the corresponding component is used.

The owner-only candidate currently shares this package with V3. Importing it
must not implicitly load the old processor, planner, and executor. The exports
remain available to the bridge and its callers during the migration.
"""

from importlib import import_module

_EXPORTS = {
    **{name: 'contracts' for name in (
        'CompletionCondition', 'FailurePolicy', 'MutationScope', 'PhaseEvidence',
        'PhaseOutcome', 'PhasePlan', 'PhaseStatus', 'PhaseSubgoal',
        'SideEffectClass', 'TacticalAction', 'TacticalState', 'new_tactical_state',
    )},
    'PhaseExecutor': 'executor',
    **{name: 'revealing' for name in (
        'Capability', 'ToolRevealDecision', 'ToolRevealPolicy', 'ToolRevealResult',
    )},
    **{name: 'outer' for name in (
        'PhaseReview', 'PhaseReviewStatus', 'StrategicAction', 'StrategicDecision',
        'V3OuterController',
    )},
    'HierarchicalCAPDProcessor': 'processor',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'.{module_name}', __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
