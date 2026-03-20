from __future__ import annotations

import pytest

from alphonse.agent.cortex.graph import CortexGraph


def test_cortex_graph_invoke_rejects_invalid_llm_client_contract() -> None:
    graph = CortexGraph()
    with pytest.raises(ValueError) as exc:
        graph.invoke(state={}, text="hello", llm_client=object())
    assert "provider_contract_error:text_completion_missing" in str(exc.value)
