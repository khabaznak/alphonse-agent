from __future__ import annotations

from alphonse.agent.cognition.narration.outbound_narration_orchestrator import _normalize_channel_type
from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.io.registry import build_default_io_registry
from alphonse.agent.io.voice_channel import VoiceExtremityAdapter


class _StubSpeaker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, *, text: str, voice: str = "default", blocking: bool = False, volume=None):  # noqa: ANN001
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "blocking": blocking,
                "volume": volume,
            }
        )
        return {"status": "ok", "result": {"mode": "non_blocking"}, "error": None, "metadata": {}}


def test_voice_extremity_delivers_using_local_speaker_tool() -> None:
    speaker = _StubSpeaker()
    adapter = VoiceExtremityAdapter(speaker=speaker)
    adapter.deliver(
        NormalizedOutboundMessage(
            message="Hola Alex",
            channel_type="voice",
            channel_target="local",
            audience={"kind": "system", "id": "system"},
            correlation_id="cid-voice-1",
            metadata={},
        )
    )
    assert len(speaker.calls) == 1
    assert speaker.calls[0]["text"] == "Hola Alex"
    assert speaker.calls[0]["blocking"] is False


def test_default_io_registry_contains_voice_extremity() -> None:
    registry = build_default_io_registry()
    assert registry.get_extremity("voice") is not None


def test_normalize_channel_type_maps_mouth_to_voice() -> None:
    assert _normalize_channel_type("mouth") == "voice"
