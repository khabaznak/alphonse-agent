from alphonse.agent.cognition.providers.llamafarm import LlamaFarmClient
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.providers.opencode import OpenCodeClient
from alphonse.agent.cognition.providers.openai import OpenAIClient


def build_provider_client(config: dict):
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
    if provider_type == "opencode":
        settings = provider_config.get("opencode", {})
        return OpenCodeClient(
            base_url=settings.get("base_url", "http://127.0.0.1:4096"),
            model=settings.get("model", "ollama/mistral:7b-instruct"),
            timeout=settings.get("timeout", 120),
            chat_path=settings.get("chat_path", "/v1/chat/completions"),
            api_key_env=settings.get("api_key_env", "OPENCODE_API_KEY"),
            username_env=settings.get("username_env", "OPENCODE_SERVER_USERNAME"),
            password_env=settings.get("password_env", "OPENCODE_SERVER_PASSWORD"),
        )
    if provider_type in {"llamafarm", "llama_farm"}:
        settings = provider_config.get("llamafarm", {})
        return LlamaFarmClient(
            base_url=settings.get("base_url", "http://127.0.0.1:8002/v1"),
            model=settings.get("model", "mistral:7b-instruct"),
            api_key_env=settings.get("api_key_env", "LLAMAFARM_API_KEY"),
            timeout=settings.get("timeout", 120),
        )

    settings = provider_config.get("ollama", {})
    return OllamaClient(
        base_url=settings.get("base_url", "http://localhost:11434"),
        model=settings.get("model", "mistral:7b-instruct"),
        timeout=settings.get("timeout", 120),
    )


def get_provider_info(config: dict) -> dict:
    mode = str(config.get("mode", "test")).lower()
    providers = config.get("providers", {})
    provider_config = providers.get(mode, {})
    provider_type = provider_config.get("type", "ollama")

    if provider_type == "openai":
        settings = provider_config.get("openai", {})
        model = settings.get("model", "gpt-4o-mini")
    elif provider_type == "opencode":
        settings = provider_config.get("opencode", {})
        model = settings.get("model", "ollama/mistral:7b-instruct")
    elif provider_type in {"llamafarm", "llama_farm"}:
        settings = provider_config.get("llamafarm", {})
        model = settings.get("model", "mistral:7b-instruct")
    else:
        settings = provider_config.get("ollama", {})
        model = settings.get("model", "mistral:7b-instruct")

    return {
        "mode": mode,
        "provider": provider_type,
        "model": model,
    }
