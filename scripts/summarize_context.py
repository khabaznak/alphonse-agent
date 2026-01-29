import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from alphonse.cognition.providers.ollama import OllamaClient
from alphonse.cognition.reasoner import REFERENCE_FILES, SUMMARY_PATH
from alphonse.config import load_alphonse_config


SYSTEM_PROMPT = (
    "Summarize the following Atrium guiding documents into a concise reference "
    "for local inference. Preserve key principles, boundaries, tone, and "
    "behavioral constraints. Keep it under 350 words."
)


def main() -> None:
    config = load_alphonse_config()
    provider_settings = (
        config.get("providers", {})
        .get("test", {})
        .get("ollama", {})
    )

    client = OllamaClient(
        base_url=provider_settings.get("base_url", "http://localhost:11434"),
        model=provider_settings.get("model", "mistral:7b-instruct"),
        timeout=provider_settings.get("timeout", 120),
    )

    documents = "\n\n".join(
        path.read_text(encoding="utf-8").strip()
        for path in REFERENCE_FILES
        if path.exists()
    )

    start = time.monotonic()
    summary = client.complete(system_prompt=SYSTEM_PROMPT, user_prompt=documents)
    elapsed = time.monotonic() - start

    SUMMARY_PATH.write_text(f"{summary.strip()}\n", encoding="utf-8")

    print(f"Summary written to {SUMMARY_PATH} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
