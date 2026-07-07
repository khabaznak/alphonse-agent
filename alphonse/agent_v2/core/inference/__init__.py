"""Inference routing contracts for Alphonse agent v2."""

from alphonse.agent_v2.core.inference.models import InferencePurpose
from alphonse.agent_v2.core.inference.models import InferenceRequest
from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.inference.models import ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceProvider
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.inference.stub import StubInferenceProvider

__all__ = [
    "InferenceProvider",
    "InferencePurpose",
    "InferenceRequest",
    "InferenceResult",
    "InferenceRouter",
    "ModelProfile",
    "StubInferenceProvider",
]
