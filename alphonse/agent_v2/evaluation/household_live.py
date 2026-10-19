"""Tool-disabled live inference transport for synthetic household runtime trials."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.core.inference.openai_codex import _render_envelope, _try_parse_json_object
from .context_pilot_live import CLI, configured_model


class HouseholdLiveProvider:
    supports_image_input = False

    def __init__(self, *, max_calls=24):
        self.model = configured_model()
        self.max_calls = max_calls
        self.calls = []

    def generate_markdown(self, request):
        return self._generate(request, structured=False)

    def generate_json(self, request):
        return self._generate(request, structured=True)

    def plan_tool_call(self, request):
        return self._generate(request, structured=True)

    def _generate(self, request, *, structured):
        if len(self.calls) >= self.max_calls:
            raise RuntimeError("household_trial_model_call_limit")
        if request.cancel_checker and request.cancel_checker():
            raise RuntimeError("household_trial_cancelled")
        field = "content"
        instruction = (
            "Return the requested JSON object directly, with no markdown fences or commentary. "
            "Follow the schema and instructions in the runtime prompt exactly."
            if structured else "Return the requested user-facing text in the content field."
        )
        payload = _render_envelope(request, instructions=instruction, tools=request.tools)
        entry = {"purpose": request.purpose.value, "prompt": request.prompt,
                 "task_id": request.task_id,
                 "tool_ids": [tool.tool_id for tool in request.tools], "status": "started"}
        self.calls.append(entry)
        started = perf_counter()
        try:
            if "PRIVATE_SENTINEL" in payload:
                raise RuntimeError("household_private_source_before_provider")
            with tempfile.TemporaryDirectory(prefix="household-inference-") as directory:
                root = Path(directory)
                instructions = root / "instructions.txt"
                instructions.write_text(
                    "You are Alphonse's inference provider for a synthetic household runtime trial. "
                    "Follow the supplied runtime prompt. Return only the requested structured output. "
                    "You cannot execute tools: Alphonse validates and executes composed calls from its fixture registry. "
                    "Use supplied task and resource paths only; your transport directory is not a task resource. "
                    + instruction
                )
                schema = root / "schema.json"
                response_schema = request.output_schema if structured else {
                    "type": "object", "additionalProperties": False,
                    "properties": {field: {"type": "string"}}, "required": [field]}
                if response_schema is not None:
                    schema.write_text(json.dumps(response_schema))
                skill_paths = set()
                for folder in (Path.home() / ".codex/skills", Path.home() / ".agents/skills"):
                    if folder.exists():
                        skill_paths.update(str(path.parent) for path in folder.rglob("SKILL.md"))
                overrides = "[" + ",".join("{path=" + json.dumps(path) + ",enabled=false}" for path in sorted(skill_paths)) + "]"
                command = [CLI, "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
                           "--skip-git-repo-check", "--sandbox", "read-only", "--model", self.model,
                           "--json"]
                if response_schema is not None:
                    command.extend(["--output-schema", str(schema)])
                for setting in (
                    "features.shell_tool=false", "features.apps=false", "features.plugins=false",
                    "features.multi_agent=false", "tools.view_image=false", "features.browser_use=false",
                    "features.image_generation=false", "features.skill_search=false", "features.hooks=false",
                    "features.memories=false", "features.sleep_tool=false", "project_doc_max_bytes=0",
                    'web_search="disabled"', "skills.max_context_tokens=1", "skills.config=" + overrides,
                    "model_instructions_file=" + json.dumps(str(instructions)),
                ):
                    command.extend(["-c", setting])
                completed = subprocess.run(command, input=payload, text=True, capture_output=True,
                                           cwd=root, timeout=120)
            if completed.returncode:
                # Raw diagnostics may contain credentials/host paths: retain only a category.
                diagnostic = (completed.stdout + completed.stderr).casefold()
                category = next((label for label, words in (
                    ("rate_limit", ("rate limit", "429")), ("authentication", ("401", "unauthorized")),
                    ("transport", ("network", "connection", "stream disconnected")),
                ) if any(word in diagnostic for word in words)), "unknown")
                raise RuntimeError("household_provider_exit:" + str(completed.returncode) + ":" + category)
            messages = []
            usage = None
            for line in completed.stdout.splitlines():
                event = json.loads(line)
                if event["type"] == "item.completed":
                    item = event["item"]
                    if item["type"] == "agent_message":
                        messages.append(item["text"])
                    elif item["type"] not in {"reasoning", "error"}:
                        raise RuntimeError("household_provider_unexpected_tool")
                elif event["type"] == "turn.completed":
                    usage = event.get("usage")
                elif event["type"] in {"turn.failed", "error"}:
                    raise RuntimeError("household_provider_turn_failed")
            if len(messages) != 1:
                raise RuntimeError("household_provider_expected_one_response")
            entry["raw_output"] = messages[0]
            value = "" if structured else json.loads(messages[0])[field]
            parsed = _try_parse_json_object(messages[0]) if structured else None
            if structured and parsed is None:
                raise RuntimeError("household_provider_invalid_json")
            entry.update(status="success", output=parsed if structured else value, usage=usage)
            return InferenceResult(content="" if structured else value, json_value=parsed,
                                   usage=usage or {}, model_profile=request.model_profile)
        except Exception as exc:
            entry.update(status="failed", error=type(exc).__name__)
            if str(exc).startswith("household_"):
                entry["error_code"] = str(exc)
            raise
        finally:
            entry["latency_ms"] = round((perf_counter() - started) * 1000, 2)
            output = entry.get("output")
            tool_id = output.get("tool_id", "") if isinstance(output, dict) else ""
            print("Inference", len(self.calls), request.purpose.value, entry["status"], tool_id, flush=True)
