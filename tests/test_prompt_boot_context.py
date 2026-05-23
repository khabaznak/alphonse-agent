from __future__ import annotations

from pathlib import Path

from alphonse.config.prompt_context import load_boot_prompt_context
from alphonse.agent.cognition import prompt_templates_runtime
from alphonse.agent.cortex.task_mode import prompt_templates


def test_boot_prompt_context_missing_files_returns_empty_sections(tmp_path: Path) -> None:
    context = load_boot_prompt_context(tmp_path)

    assert context == {
        "PHILOSOPHY_SECTION": "",
        "CORE_CONTEXT_SECTION": "",
    }


def test_boot_prompt_context_blank_files_return_empty_sections(tmp_path: Path) -> None:
    (tmp_path / "Philosophy.md").write_text("\n \n", encoding="utf-8")
    (tmp_path / "CoreContext.md").write_text("\t\n", encoding="utf-8")

    context = load_boot_prompt_context(tmp_path)

    assert context["PHILOSOPHY_SECTION"] == ""
    assert context["CORE_CONTEXT_SECTION"] == ""


def test_boot_prompt_context_loads_trimmed_content(tmp_path: Path) -> None:
    (tmp_path / "Philosophy.md").write_text("\nAct with restraint.\n\n", encoding="utf-8")
    (tmp_path / "CoreContext.md").write_text("\nHousehold context.\n\n", encoding="utf-8")

    context = load_boot_prompt_context(tmp_path)

    assert context["PHILOSOPHY_SECTION"] == "Act with restraint."
    assert context["CORE_CONTEXT_SECTION"] == "Household context."


def test_boot_prompt_context_strips_matching_top_level_headings(tmp_path: Path) -> None:
    (tmp_path / "Philosophy.md").write_text(
        "# Philosophy\n\nAct with restraint.\n",
        encoding="utf-8",
    )
    (tmp_path / "CoreContext.md").write_text(
        "# Core Context\n\nHousehold context.\n",
        encoding="utf-8",
    )

    context = load_boot_prompt_context(tmp_path)

    assert context["PHILOSOPHY_SECTION"] == "Act with restraint."
    assert context["CORE_CONTEXT_SECTION"] == "Household context."


def test_next_step_system_prompt_injects_boot_context(monkeypatch) -> None:
    monkeypatch.setattr(
        prompt_templates,
        "load_boot_prompt_context",
        lambda: {
            "PHILOSOPHY_SECTION": "Act with restraint.",
            "CORE_CONTEXT_SECTION": "Household context.",
        },
    )

    rendered = prompt_templates._load_next_step_system_prompt()

    assert rendered.startswith(
        "# Philosophy\n\n"
        "Act with restraint.\n\n"
        "# Core Context\n\n"
        "Household context.\n\n"
        "You are Alphonse"
    )
    assert "an iterative tool-using agent in a classic Deming's PDCA loop" in rendered


def test_next_step_system_prompt_omits_empty_boot_context(monkeypatch) -> None:
    monkeypatch.setattr(
        prompt_templates,
        "load_boot_prompt_context",
        lambda: {
            "PHILOSOPHY_SECTION": "",
            "CORE_CONTEXT_SECTION": "",
        },
    )

    rendered = prompt_templates._load_next_step_system_prompt()

    assert rendered.startswith("You are Alphonse")
    assert "# Philosophy" not in rendered
    assert "# Core Context" not in rendered
    assert "an iterative tool-using agent in a classic Deming's PDCA loop" in rendered


def test_check_judge_system_prompt_injects_boot_context(monkeypatch) -> None:
    monkeypatch.setattr(
        prompt_templates_runtime,
        "load_boot_prompt_context",
        lambda: {
            "PHILOSOPHY_SECTION": "Act with restraint.",
            "CORE_CONTEXT_SECTION": "Household context.",
        },
    )

    rendered = prompt_templates_runtime._seed_text_with_boot_context(
        "pdca.check.judge.system.j2",
    )

    assert rendered.startswith(
        "# Philosophy\n\n"
        "Act with restraint.\n\n"
        "# Core Context\n\n"
        "Household context.\n\n"
        "# ROLE"
    )
    assert "You are Alphonse the Judge stage (Check)" in rendered


def test_check_judge_system_prompt_omits_empty_boot_context(monkeypatch) -> None:
    monkeypatch.setattr(
        prompt_templates_runtime,
        "load_boot_prompt_context",
        lambda: {
            "PHILOSOPHY_SECTION": "",
            "CORE_CONTEXT_SECTION": "",
        },
    )

    rendered = prompt_templates_runtime._seed_text_with_boot_context(
        "pdca.check.judge.system.j2",
    )

    assert rendered.startswith("# ROLE")
    assert "# Philosophy" not in rendered
    assert "# Core Context" not in rendered
    assert "You are Alphonse the Judge stage (Check)" in rendered
