"""Compatibility import for the shared candidate inference adapter.

Evaluation and runtime hosts must both supply an explicit task identity; this
module does not register the candidate for household use.
"""
from alphonse.agent_v2.core.intelligence.v3.router_candidate_chooser import RouterCandidateChooser

__all__ = ['RouterCandidateChooser']
