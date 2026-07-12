from __future__ import annotations

import pytest

from alphonse.agent_v2.agent_config import AgentConfigPromptLoader
from alphonse.agent_v2.agent_config import AgentConfigStore
from alphonse.agent_v2.agent_config import GLOBAL_CONTEXT_FILE
from alphonse.agent_v2.agent_config import PHILOSOPHY_FILE
from alphonse.agent_v2.core.intelligence import TaskState
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import plan_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import _render_acceptance_criteria_prompt
from alphonse.agent_v2.core.intelligence.pdca.nodes.check_node import _render_criteria_review_prompt
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import _render_tool_call_plan_prompt


def test_agent_config_store_seeds_defaults_and_persists_edits(tmp_path) -> None:
    store = AgentConfigStore(tmp_path / "agent-config")

    documents = store.list_documents()
    saved = store.save(PHILOSOPHY_FILE, "# Philosophy\n\nBe concise.\n")

    assert [document.file_name for document in documents] == [GLOBAL_CONTEXT_FILE, PHILOSOPHY_FILE]
    assert saved.content == "# Philosophy\n\nBe concise.\n"
    assert store.read(PHILOSOPHY_FILE).content == saved.content


def test_agent_config_store_rejects_unknown_file_without_writing(tmp_path) -> None:
    store = AgentConfigStore(tmp_path / "agent-config")
    original = store.read(GLOBAL_CONTEXT_FILE).content

    with pytest.raises(ValueError, match="agent_config_file_not_allowed"):
        store.save("unknown.md", "bad")

    assert store.read(GLOBAL_CONTEXT_FILE).content == original


def test_agent_prompt_loader_is_a_startup_snapshot(tmp_path) -> None:
    store = AgentConfigStore(tmp_path / "agent-config")
    store.save(PHILOSOPHY_FILE, "first")
    loader = AgentConfigPromptLoader.from_store(store)
    store.save(PHILOSOPHY_FILE, "second")

    assert loader.load(PHILOSOPHY_FILE).content == "first"
    assert AgentConfigPromptLoader.from_store(store).load(PHILOSOPHY_FILE).content == "second"


def test_capd_prompt_templates_accept_agent_configuration() -> None:
    task = TaskState(goal="Respond", acceptance_criteria_md="1.- [ ] Reply")
    common = {
        "philosophy_md": "Act with care.",
        "global_context_md": "Global context.",
        "user_context_md": "User context.",
        "project_context_md": "Project context.",
    }

    for prompt in (
        _render_tool_call_plan_prompt(task, (), **common),
        _render_acceptance_criteria_prompt(task, **common),
        _render_criteria_review_prompt(task, {}, **common),
    ):
        assert prompt.index("## Philosophy.md") < prompt.index("## GlobalContext.md")
        assert prompt.index("## GlobalContext.md") < prompt.index("## User Context")
        assert prompt.index("## User Context") < prompt.index("## Project Context")
        for sentinel in common.values():
            assert sentinel in prompt


def test_plan_node_reads_context_from_the_runtime_prompt_snapshot(tmp_path) -> None:
    store = AgentConfigStore(tmp_path / "agent-config")
    store.save(PHILOSOPHY_FILE, "Snapshot philosophy")
    store.save(GLOBAL_CONTEXT_FILE, "Snapshot global context")
    task = TaskState(goal="Respond", acceptance_criteria_md="1.- [ ] Reply")

    plan_node(
        task,
        CoreLoopContext(messages=InMemoryMessageQueue(), prompts=AgentConfigPromptLoader.from_store(store)),
    )

    assert "Snapshot philosophy" in task.metadata["tool_call_plan_prompt"]
    assert "Snapshot global context" in task.metadata["tool_call_plan_prompt"]
