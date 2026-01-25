from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rex.cognition.providers.ollama import OllamaClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILES = (
    PROJECT_ROOT / "CONSTITUTION.md",
    PROJECT_ROOT / "rex" / "identity.md",
    PROJECT_ROOT / "rex" / "tone.md",
    PROJECT_ROOT / "rex" / "startup.md",
    PROJECT_ROOT / "docs" / "vision.md",
    PROJECT_ROOT / "docs" / "philosophy.md",
    PROJECT_ROOT / "docs" / "decisions.md",
)


@dataclass(frozen=True)
class ReasoningInput:
    snapshot: dict


def build_system_prompt() -> str:
    references = "\n\n".join(_read_reference_file(path) for path in REFERENCE_FILES)
    return (
        "You are Rex, the resident mayordomo of Atrium. "
        "Use the constitution and guiding documents to respond with composure, "
        "restraint, and clarity. Avoid commands; offer grounded observations.\n\n"
        "Guiding documents:\n"
        f"{references}"
    )


def build_user_prompt(payload: ReasoningInput, instruction: str | None = None) -> str:
    directive = instruction or "Provide a concise response for the household."
    return (
        "Current awareness snapshot:\n"
        f"{payload.snapshot}\n\n"
        f"{directive}"
    )


def build_reasoning_prompt(payload: ReasoningInput) -> str:
    return f"{build_system_prompt()}\n\n{build_user_prompt(payload)}"


class Reasoner:
    def __init__(self, client: OllamaClient):
        self._client = client

    def reason(self, instruction: str, snapshot: dict) -> str:
        payload = ReasoningInput(snapshot=snapshot)
        return self._client.complete(
            system_prompt=build_system_prompt(),
            user_prompt=build_user_prompt(payload, instruction=instruction),
        )


def interpret_status(payload: ReasoningInput, llm_call: Callable[[str], str]) -> str:
    prompt = build_reasoning_prompt(payload)
    return llm_call(prompt)


def interpret_status_with_ollama(
    payload: ReasoningInput,
    client: OllamaClient | None = None,
) -> str:
    ollama_client = client or OllamaClient()
    return ollama_client.complete(
        system_prompt=build_system_prompt(),
        user_prompt=build_user_prompt(payload),
    )


def _read_reference_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return f"[Missing file: {path.name}]"
