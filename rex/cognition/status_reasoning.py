from core.context.awareness import get_awareness_snapshot
from rex.cognition.providers.ollama import OllamaClient
from rex.cognition.providers.openai import OpenAIClient
from rex.config import load_rex_config


STATUS_SYSTEM_PROMPT = (
    "You are Rex, a calm and restrained domestic presence.\n\n"
    "Given the following system snapshot, summarize the current state calmly in one sentence.\n"
    "Do not suggest actions.\n"
    "Do not speculate.\n"
    "Keep it under 2 sentences."
)


def reason_about_status():
    snapshot = get_awareness_snapshot()
    config = load_rex_config()
    client = _build_client(config)

    message = client.complete(
        system_prompt=STATUS_SYSTEM_PROMPT,
        user_prompt=str(snapshot),
    )

    return message, snapshot


def _build_client(config: dict):
    mode = str(config.get("mode", "test")).lower()
    providers = config.get("providers", {})
    provider_config = providers.get(mode, {})
    provider_type = provider_config.get("type", "ollama")

    if provider_type == "openai":
        settings = provider_config.get("openai", {})
        return OpenAIClient(
            base_url=settings.get("base_url", "https://api.openai.com/v1"),
            model=settings.get("model", "gpt-4o-mini"),
            api_key_env=settings.get("api_key_env", "OPENAI_API_KEY"),
            timeout=settings.get("timeout", 60),
        )

    settings = provider_config.get("ollama", {})
    return OllamaClient(
        base_url=settings.get("base_url", "http://localhost:11434"),
        model=settings.get("model", "mistral:7b-instruct"),
        timeout=settings.get("timeout", 120),
    )
