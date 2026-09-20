"""Offline evaluation helpers for comparing Alphonse intelligence engines."""

from typing import Any

__all__ = ["EngineTrace", "EvaluationCase", "ReplayHarness", "load_corpus"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from alphonse.agent_v2.evaluation import replay
    return getattr(replay, name)
