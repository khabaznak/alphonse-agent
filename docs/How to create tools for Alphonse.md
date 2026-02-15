# How to Create Tools for Alphonse

This guide explains the standard way to add a new tool to Alphonse so it can be safely used by planning and execution flows.

## 1) Implement the Tool

Create a file in:

- `alphonse/agent/tools/`

Typical pattern:

- Class with `execute(...) -> dict[str, Any]`
- Deterministic return shape
- `status: "ok"` on success, `status: "failed"` on failure
- Structured error codes (for planner reasoning)

Example skeleton:

```python
from __future__ import annotations

from typing import Any


class MyTool:
    def execute(self, *, input_value: str) -> dict[str, Any]:
        value = str(input_value or "").strip()
        if not value:
            return {"status": "failed", "error": "input_required"}
        # do work...
        return {"status": "ok", "result": "..."}
```

## 2) Register the Tool in Runtime Registry

Edit:

- `alphonse/agent/tools/registry.py`

Steps:

1. Import your tool class.
2. Instantiate it inside `build_default_tool_registry()`.
3. Register it with a unique tool key:
   - `registry.register("my_tool_name", my_tool)`

## 3) Expose Tool Schema to Tool-Calling LLM

Edit:

- `alphonse/agent/cognition/tool_schemas.py`

Add a JSON function schema entry with:

- `name`
- `description`
- `parameters` (JSON Schema)
- `required` fields

Important:

- Schema and runtime registration key must match exactly.

## 4) Add Tool Description to Prompt Tool Catalog

Edit:

- `alphonse/agent/cognition/prompt_seeds/planning.tools.md.j2`

Add a section describing:

- when to use
- expected inputs
- return behavior

Note:

- This markdown is not auto-generated from `tool_schemas.py`.
- Keep both files in sync manually.

## 5) Wire Special Execution Logic (If Needed)

If your tool needs custom handling, edit:

- `alphonse/agent/cortex/nodes/plan.py`

Usually in `_execute_tool_step(...)`:

- invoke the tool
- map result into standard outcome dict
- preserve structured failure facts for planner follow-up

## 6) Export (Optional)

If needed for imports, update:

- `alphonse/agent/tools/__init__.py`

## 7) Add Tests

Add unit tests under:

- `tests/test_<tool_name>.py`

Recommended coverage:

1. Success path
2. Failure path(s) with error code checks
3. Planner integration path (if exposed to tool-calling flow)

## 8) Run Validation

Run targeted tests first:

```bash
pytest -q tests/test_<tool_name>.py
```

Then run relevant planner/integration tests:

```bash
pytest -q tests/test_plan_tool_call_loop.py
```

## 9) Safety and Design Rules

1. Do not expose raw host filesystem paths to the LLM.
2. Keep permissions/policy checks outside of the model.
3. Return machine-readable failures (`error`, optional `retryable`) so planner can recover.
4. Avoid side effects unless explicitly required.
5. Keep tool I/O small and explicit.

## 10) Quick Checklist

- [ ] Tool class added in `alphonse/agent/tools/`
- [ ] Registered in `build_default_tool_registry()`
- [ ] Added in `tool_schemas.py`
- [ ] Added in `planning.tools.md.j2`
- [ ] Planner wiring updated (if needed)
- [ ] Tests added and passing
- [ ] Logs avoid sensitive values
