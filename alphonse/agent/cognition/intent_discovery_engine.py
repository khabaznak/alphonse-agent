from __future__ import annotations

import json
import logging
import re
from typing import Any

from pathlib import Path

from alphonse.agent.cognition.prompt_store import PromptContext, SqlitePromptStore
from alphonse.agent.io import get_io_registry
from alphonse.agent.nervous_system.tool_configs import get_active_tool_config

logger = logging.getLogger(__name__)


_DISSECTOR_KEY = "intent_discovery.dissector.v1"
_VISIONARY_KEY = "intent_discovery.visionary.v1"
_PLANNER_KEY = "intent_discovery.planner.v1"


_DISSECTOR_SYSTEM_FALLBACK = (
    "You are Alphonse, a message dissection engine for a personal AI assistant. "
    "Your job is to split the user message into chunks, each with a single clear intention. "
    "Output valid JSON only. No markdown. No explanations."
)

_DISSECTOR_USER_FALLBACK = (
    "Rules:\n"
    "- Just dissect structure; do not solve or execute.\n"
    "- If the message contains multiple intents, return an array with multiple objects.\n"
    "- Keep strings short. Prefer null for missing fields.\n"
    "- If unsure, lower confidence but still output valid JSON.\n"
    "- Use verb names for actions (function-like).\n\n"
    "Return JSON with this shape:\n"
    "[\n"
    "  {\n"
    "    \"action\": \"<verb>\",\n"
    "    \"chunk\": \"<short text>\",\n"
    "    \"intention\": \"<short intention>\",\n"
    "    \"confidence\": \"low|medium|high\"\n"
    "  }\n"
    "]\n\n"
    "Message:\n<<<\n{MESSAGE_TEXT}\n>>>"
)

_VISIONARY_SYSTEM_FALLBACK = (
    "You are Alphonse, a mission designer. "
    "Given one chunk and its intention, produce acceptance criteria for success. "
    "Output valid JSON only. No markdown. No explanations."
)

_VISIONARY_USER_FALLBACK = (
    "Chunk:\n{CHUNK_TEXT}\n\n"
    "Intention:\n{INTENTION}\n\n"
    "Return JSON:\n{\n  \"acceptanceCriteria\": [\"...\"]\n}"
)

_PLANNER_SYSTEM_FALLBACK = (
    "You are Alphonse, a master tool user. "
    "Your mission is to produce the shortest, best execution plan to accomplish the user's objective. "
    "Output valid JSON only. No markdown. No explanations."
)

_PLANNER_USER_FALLBACK = (
    "Rules:\n"
    "- Split the message into components and keep each component short.\n"
    "- For each component, infer a single intention.\n"
    "- Create an executionPlan as ordered tool calls. Include only necessary calls.\n"
    "- If a tool call depends on missing data, include a question using askQuestion.\n"
    "- If a tool requires parameters you do not have, do NOT emit that tool call. "
    "Instead emit askQuestion for the missing parameter(s).\n"
    "- Tool names must match exactly as listed in AVAILABLE TOOLS.\n"
    "- Output schema:\n"
    "  {\"executionPlan\": [{\"tool\": \"...\", \"parameters\": { ... }, \"executed\": false}]}\n\n"
    "Chunk:\n{CHUNK_TEXT}\n\n"
    "Intention:\n{INTENTION}\n\n"
    "Acceptance criteria:\n{ACCEPTANCE_CRITERIA}\n\n"
    "AVAILABLE TOOLS:\n{AVAILABLE_TOOLS}\n"
)

_DISSECTOR_GUARDRAILS = (
    "Hard constraints:\n"
    "- If the user message contains one goal, return exactly one chunk.\n"
    "- Do not split a single an action chunk into separate chunks for action/time/purpose.\n"
)

_PLANNER_GUARDRAILS = (
    "Hard constraints:\n"
    "- Never invent tool names. Use only tools that appear exactly in AVAILABLE TOOLS.\n"
    "- If no single listed tool can achieve the goal, produce the shortest valid multi-step sequence of listed tool calls that satisfies the acceptance criteria.\n"
    "- Never ask the user to choose or confirm internal tool/function names.\n"
    "- Ask questions only for missing end-user data, not implementation choices.\n"
    "- Prefer the shortest plan (one step when enough information is present).\n"
)


def discover_plan(
    *,
    text: str,
    llm_client: object | None,
    available_tools: str,
    locale: str | None = None,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Return a dict with discovered execution plans per chunk.

    Output shape:
    {
      "chunks": [
        {"chunk": "...", "intention": "...", "confidence": "low"},
      ],
      "plans": [
        {"chunk_index": 0, "acceptanceCriteria": [...], "executionPlan": [...]},
      ]
    }
    """
    if not llm_client:
        return {"chunks": [], "plans": []}

    strategy = (strategy or get_discovery_strategy()).strip().lower()
    store = SqlitePromptStore()
    context = PromptContext(
        locale=locale or "any",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
    )

    if strategy == "single_pass":
        return _single_pass_plan(
            text=text,
            llm_client=llm_client,
            available_tools=available_tools,
            store=store,
            context=context,
        )

    dissector_system = _get_template(
        store, _DISSECTOR_KEY, context, _DISSECTOR_SYSTEM_FALLBACK
    )
    dissector_user = _render_template(
        _get_template(
            store,
            _DISSECTOR_KEY + ".user",
            context,
            _DISSECTOR_USER_FALLBACK,
        ),
        {"MESSAGE_TEXT": text},
    )
    dissector_user = f"{dissector_user}\n\n{_DISSECTOR_GUARDRAILS}"

    raw_chunks = _call_llm(llm_client, dissector_system, dissector_user)
    chunks = _parse_chunks(raw_chunks)

    plans: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        chunk_text = str(chunk.get("chunk") or "")
        intention = str(chunk.get("intention") or "")

        visionary_system = _get_template(store, _VISIONARY_KEY, context, _VISIONARY_SYSTEM_FALLBACK)
        visionary_user = _render_template(
            _get_template(store, _VISIONARY_KEY + ".user", context, _VISIONARY_USER_FALLBACK),
            {"CHUNK_TEXT": chunk_text, "INTENTION": intention},
        )
        raw_acceptance = _call_llm(llm_client, visionary_system, visionary_user)
        acceptance = _parse_acceptance(raw_acceptance)

        planner_system = _get_template(store, _PLANNER_KEY, context, _PLANNER_SYSTEM_FALLBACK)
        planner_user = _render_template(
            _get_template(store, _PLANNER_KEY + ".user", context, _PLANNER_USER_FALLBACK),
            {
                "CHUNK_TEXT": chunk_text,
                "INTENTION": intention,
                "ACCEPTANCE_CRITERIA": json.dumps(acceptance, ensure_ascii=False),
                "AVAILABLE_TOOLS": available_tools,
            },
        )
        planner_user = f"{planner_user}\n\n{_PLANNER_GUARDRAILS}"
        raw_plan = _call_llm(llm_client, planner_system, planner_user)
        execution_plan = _parse_execution_plan(raw_plan)
        plans.append(
            {
                "chunk_index": idx,
                "acceptanceCriteria": acceptance,
                "executionPlan": execution_plan,
            }
        )

    return {"chunks": chunks, "plans": plans}


def _single_pass_plan(
    *,
    text: str,
    llm_client: object,
    available_tools: str,
    store: SqlitePromptStore,
    context: PromptContext,
) -> dict[str, Any]:
    planner_system = _get_template(store, _PLANNER_KEY, context, _PLANNER_SYSTEM_FALLBACK)
    planner_user = _render_template(
        _get_template(store, _PLANNER_KEY + ".user", context, _PLANNER_USER_FALLBACK),
        {
            "CHUNK_TEXT": text,
            "INTENTION": "overall",
            "ACCEPTANCE_CRITERIA": "[]",
            "AVAILABLE_TOOLS": available_tools,
        },
    )
    planner_user = f"{planner_user}\n\n{_PLANNER_GUARDRAILS}"
    raw_plan = _call_llm(llm_client, planner_system, planner_user)
    execution_plan = _parse_execution_plan(raw_plan)
    return {
        "chunks": [{"chunk": text, "intention": "overall", "confidence": "medium"}],
        "plans": [
            {
                "chunk_index": 0,
                "acceptanceCriteria": [],
                "executionPlan": execution_plan,
            }
        ],
    }


def get_discovery_strategy(default: str = "multi_pass") -> str:
    config = get_active_tool_config("routing_strategy")
    if not config or not isinstance(config.get("config"), dict):
        return default
    raw = config.get("config") or {}
    strategy = str(raw.get("strategy") or "").strip()
    return strategy or default


def format_available_abilities() -> str:
    specs = _load_ability_specs()
    lines: list[str] = [
        "- askQuestion(question:string, slot?:string, bind?:object): Ask for missing end-user data only. Never ask about internal tool names.",
    ]
    for spec in specs:
        intent = str(spec.get("intent_name") or "").strip()
        if not intent:
            continue
        params = spec.get("input_parameters") if isinstance(spec.get("input_parameters"), list) else []
        summary = _ability_summary(spec)
        if params:
            params_desc = ", ".join(
                _render_param_signature(p) for p in params if p.get("name")
            )
            lines.append(f"- {intent}({params_desc}) -> {summary}")
        else:
            lines.append(f"- {intent}() -> {summary}")
    return "\n".join(lines)


def format_available_ability_catalog() -> str:
    specs = _load_ability_specs()
    tools: list[dict[str, Any]] = [
        {
            "tool": "askQuestion",
            "summary": "Ask the user for missing parameters to continue.",
            "intents_covered": ["clarification", "slot_fill"],
            "examples": ["Ask only for missing user data."],
            "input_parameters": [
                {"name": "question", "type": "string", "required": True},
                {"name": "slot", "type": "string", "required": False},
                {"name": "bind", "type": "object", "required": False},
            ],
        }
    ]
    for spec in specs:
        tool = str(spec.get("intent_name") or "").strip()
        if not tool:
            continue
        raw_params = (
            spec.get("input_parameters")
            if isinstance(spec.get("input_parameters"), list)
            else []
        )
        params: list[dict[str, Any]] = []
        for item in raw_params:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            params.append(
                {
                    "name": name,
                    "type": str(item.get("type") or "string"),
                    "required": bool(item.get("required", False)),
                }
            )
        tools.append(
            {
                "tool": tool,
                "summary": _ability_summary(spec),
                "intents_covered": _ability_intents_covered(spec),
                "examples": _ability_examples(spec),
                "actions": _ability_actions(spec),
                "required_parameters": [
                    p["name"] for p in params if bool(p.get("required"))
                ],
                "input_parameters": params,
                "uses_tools": [
                    str(item).strip()
                    for item in (spec.get("tools") if isinstance(spec.get("tools"), list) else [])
                    if str(item).strip()
                ],
            }
        )
    return json.dumps(
        {
            "tools": tools,
            "io_channels": _io_channel_catalog(),
        },
        ensure_ascii=False,
    )


def _call_llm(llm_client: object, system_prompt: str, user_prompt: str) -> str:
    try:
        return str(llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception as exc:
        logger.warning("intent discovery LLM call failed: %s", exc)
        return ""


def _ability_summary(spec: dict[str, Any]) -> str:
    kind = str(spec.get("kind") or "").strip()
    steps = spec.get("step_sequence") if isinstance(spec.get("step_sequence"), list) else []
    if steps:
        first = steps[0]
        if isinstance(first, dict):
            action = str(first.get("action") or "").strip()
            if action:
                return f"{kind or 'ability'} via action {action}"
    return kind or "ability"


def _ability_intents_covered(spec: dict[str, Any]) -> list[str]:
    intent = str(spec.get("intent_name") or "").strip()
    if not intent:
        return []
    aliases = [intent]
    parts = intent.split(".")
    if len(parts) > 1:
        aliases.append(parts[-1])
    return aliases


def _ability_examples(spec: dict[str, Any]) -> list[str]:
    prompts = spec.get("prompts") if isinstance(spec.get("prompts"), dict) else {}
    outputs = spec.get("outputs") if isinstance(spec.get("outputs"), dict) else {}
    step_sequence = spec.get("step_sequence") if isinstance(spec.get("step_sequence"), list) else []
    examples: list[str] = []
    if prompts:
        examples.append(
            "Prompts: " + ", ".join(sorted(str(k) for k in prompts.keys()))
        )
    if outputs:
        examples.append(
            "Outputs: " + ", ".join(sorted(str(k) for k in outputs.keys()))
        )
    if step_sequence:
        action_names = [
            str(step.get("action") or "").strip()
            for step in step_sequence
            if isinstance(step, dict) and str(step.get("action") or "").strip()
        ]
        if action_names:
            examples.append("Actions: " + ", ".join(action_names))
    if not examples:
        examples.append("No explicit examples provided.")
    return examples


def _ability_actions(spec: dict[str, Any]) -> list[str]:
    step_sequence = spec.get("step_sequence") if isinstance(spec.get("step_sequence"), list) else []
    actions: list[str] = []
    for step in step_sequence:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip()
        if action:
            actions.append(action)
    return actions


def _render_param_signature(param: dict[str, Any]) -> str:
    name = str(param.get("name") or "").strip()
    ptype = str(param.get("type") or "string").strip()
    required = bool(param.get("required", False))
    suffix = "" if required else "?"
    return f"{name}{suffix}:{ptype}"


def _io_channel_catalog() -> dict[str, Any]:
    try:
        registry = get_io_registry()
        senses = sorted(str(name) for name in registry.senses.keys())
        extremities = sorted(str(name) for name in registry.extremities.keys())
    except Exception:
        senses = []
        extremities = []
    return {
        "senses": senses,
        "extremities": extremities,
        "notes": [
            "senses = inbound channels that read/normalize external input",
            "extremities = outbound channels that deliver responses/actions",
        ],
    }


def _get_template(store: SqlitePromptStore, key: str, context: PromptContext, fallback: str) -> str:
    match = store.get_template(key, context)
    if match and match.template:
        return str(match.template)
    return fallback


def _render_template(template: str, variables: dict[str, Any]) -> str:
    rendered = template
    for name, value in variables.items():
        rendered = rendered.replace("{" + name + "}", str(value))
    return rendered


def _parse_chunks(raw: str) -> list[dict[str, Any]]:
    payload = _parse_json(raw)
    if isinstance(payload, list):
        return [_normalize_chunk(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            return [_normalize_chunk(item) for item in chunks if isinstance(item, dict)]
    return []


def _parse_acceptance(raw: str) -> list[str]:
    payload = _parse_json(raw)
    if isinstance(payload, dict):
        criteria = payload.get("acceptanceCriteria")
        if isinstance(criteria, list):
            return [str(item) for item in criteria if item]
    if isinstance(payload, list):
        return [str(item) for item in payload if item]
    return []


def _parse_execution_plan(raw: str) -> list[dict[str, Any]]:
    payload = _parse_json(raw)
    if isinstance(payload, dict):
        plan = payload.get("executionPlan")
        if isinstance(plan, list):
            return [item for item in plan if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _parse_json(raw: str) -> Any:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    salvaged = _extract_first_json(candidate)
    if salvaged is None:
        return None
    sanitized = re.sub(r",\s*([}\]])", r"\1", salvaged)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        return None


def _extract_first_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalize_chunk(item: dict[str, Any]) -> dict[str, Any]:
    chunk_text = str(item.get("chunk") or "").strip()
    action = str(item.get("action") or "").strip()
    intention = str(item.get("intention") or "").strip()
    if not chunk_text and action:
        chunk_text = action
    if not intention and action:
        intention = action
    normalized = dict(item)
    normalized["chunk"] = chunk_text
    normalized["intention"] = intention
    return normalized


def _load_ability_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    specs.extend(_load_specs_file())
    specs.extend(_load_specs_db())
    unique: dict[str, dict[str, Any]] = {}
    for spec in specs:
        intent = str(spec.get("intent_name") or "").strip()
        if not intent:
            continue
        unique[intent] = spec
    return [unique[key] for key in sorted(unique.keys())]


def _load_specs_file() -> list[dict[str, Any]]:
    path = (
        Path(__file__).resolve().parents[2]
        / "nervous_system"
        / "resources"
        / "ability_specs.seed.json"
    )
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("ability spec load failed path=%s error=%s", path, exc)
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _load_specs_db() -> list[dict[str, Any]]:
    try:
        from alphonse.agent.cognition.abilities.store import AbilitySpecStore

        return AbilitySpecStore().list_enabled_specs()
    except Exception as exc:
        logger.warning("ability spec db load failed error=%s", exc)
        return []
