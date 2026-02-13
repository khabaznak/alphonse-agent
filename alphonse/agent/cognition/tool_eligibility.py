from __future__ import annotations

import re

_CAPABILITY_PATTERN = re.compile(
    r"(what can you do|capabilit|supported|features?|help|puedes hacer|capacidades?)",
    flags=re.IGNORECASE,
)
_AGENT_FACTS_PATTERN = re.compile(
    r"(agent|status|state|uptime|version|timezone|tu estado|tu nombre|qu[ié]n eres)",
    flags=re.IGNORECASE,
)


def is_tool_eligible(*, tool_name: str, user_message: str) -> tuple[bool, str | None]:
    tool = str(tool_name or "").strip()
    text = str(user_message or "").strip()
    if not tool:
        return False, "missing_tool_name"
    if tool == "meta.capabilities":
        if _CAPABILITY_PATTERN.search(text):
            return True, None
        return False, "tool_not_eligible_for_intent"
    if tool == "facts.agent.get":
        if _AGENT_FACTS_PATTERN.search(text):
            return True, None
        return False, "tool_not_eligible_for_intent"
    return True, None
