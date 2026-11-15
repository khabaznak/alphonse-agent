from __future__ import annotations

import json

import pytest

from alphonse.agent_v2.core.intelligence.task_state import TaskState


































def test_legacy_task_state_dictionary_loads_with_v2_defaults() -> None:
    restored = TaskState.from_dict({"goal": "legacy"})

    assert restored.intelligence_engine == "tactical_v2"
    assert restored.intelligence_schema_version == 2
    assert restored.hierarchical_state == {}
