from __future__ import annotations

from alphonse.agent.cortex.nodes import plan as plan_module


def test_planning_context_renders_retrieved_scratchpad_note_blocks() -> None:
    rendered = plan_module._render_context_markdown(
        {
            "facts": {
                "tool_results": [
                    {
                        "tool": "scratchpad_read",
                        "result": {
                            "doc_id": "sp_2043",
                            "title": "Weekly household chores",
                            "scope": "household",
                            "tags": ["chores", "family"],
                            "updated_at": "2026-02-17T09:15:00-06:00",
                            "mode": "tail",
                            "max_chars": 6000,
                            "content": "Laundry and dishes",
                        },
                    }
                ]
            }
        }
    )
    assert "### Retrieved Scratchpad Note (working context; may include drafts)" in rendered
    assert "- doc_id: sp_2043" in rendered
    assert "- title: Weekly household chores" in rendered
    assert "- scope: household" in rendered
    assert "- tags: chores, family" in rendered
    assert "- mode: tail (max_chars=6000)" in rendered

