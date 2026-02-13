from __future__ import annotations

from alphonse.agent.cognition.provider_selector import (
    build_provider_client,
    get_provider_info,
)
from alphonse.agent.cognition.providers.llamafarm import LlamaFarmClient
from alphonse.agent.cognition.providers.opencode import OpenCodeClient


def test_provider_selector_builds_opencode_client() -> None:
    config = {
        "mode": "test",
        "providers": {
            "test": {
                "type": "opencode",
                "opencode": {
                    "base_url": "http://127.0.0.1:4096",
                    "model": "ollama/mistral:7b-instruct",
                },
            }
        },
    }
    client = build_provider_client(config)
    assert isinstance(client, OpenCodeClient)
    info = get_provider_info(config)
    assert info["provider"] == "opencode"
    assert info["model"] == "ollama/mistral:7b-instruct"


def test_provider_selector_builds_llamafarm_client() -> None:
    config = {
        "mode": "test",
        "providers": {
            "test": {
                "type": "llamafarm",
                "llamafarm": {
                    "base_url": "http://127.0.0.1:8002/v1",
                    "model": "mistral:7b-instruct",
                },
            }
        },
    }
    client = build_provider_client(config)
    assert isinstance(client, LlamaFarmClient)
    info = get_provider_info(config)
    assert info["provider"] == "llamafarm"
    assert info["model"] == "mistral:7b-instruct"
