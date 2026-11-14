"""Intelligence processor package; legacy PDCA is loaded only on request."""

from alphonse.agent_v2.core.intelligence.task_state import TaskState

__all__ = [
    "PDCAIntelligenceProcessor",
    "TaskState",
    "build_pdca_graph",
    "run_pdca_once",
]


def __getattr__(name: str):
    if name in {"PDCAIntelligenceProcessor", "build_pdca_graph", "run_pdca_once"}:
        from alphonse.agent_v2.core.intelligence import pdca
        return getattr(pdca, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
