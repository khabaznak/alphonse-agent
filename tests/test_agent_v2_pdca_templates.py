from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader


TEMPLATE_DIR = Path("alphonse/agent_v2/core/intelligence/templates")


def test_pdca_prompt_templates_render_expected_sections() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    for template_name in ("check_prompt.j2", "plan_prompt.j2", "act_prompt.j2"):
        rendered = env.get_template(template_name).render(
            system_prompt="System instructions",
            philosophy_md="Philosophy content",
            global_context_md="Global context",
            user_personality_md="User personality",
            project_context_md="Project context",
            user_prompt="User request",
            task_state_md="Task state markdown",
        )

        assert "# System Prompt" in rendered
        assert "System instructions" in rendered
        assert "## Philosophy.md" in rendered
        assert "## GlobalContext.md" in rendered
        assert "## UserPersonality" in rendered
        assert "## Project Context" in rendered
        assert "# User Prompt" in rendered
        assert "User request" in rendered
        assert "# Task State" in rendered
        assert "Task state markdown" in rendered


def test_pdca_prompt_templates_have_stub_defaults() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    rendered = env.get_template("check_prompt.j2").render()

    assert "Stub check-node system prompt." in rendered
    assert "- (not loaded)" in rendered
    assert "- (not provided)" in rendered
