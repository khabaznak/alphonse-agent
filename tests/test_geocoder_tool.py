from __future__ import annotations

import json

import pytest

from alphonse.agent.tools.geocoder import GoogleGeocoderTool


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_google_geocoder_parses_location(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "status": "OK",
        "results": [
            {
                "formatted_address": "Guadalajara, Jalisco, Mexico",
                "place_id": "place-123",
                "types": ["locality"],
                "geometry": {"location": {"lat": 20.67, "lng": -103.35}},
            }
        ],
    }

    def _fake_urlopen(req, timeout=10):
        _ = req, timeout
        return _FakeResponse(payload)

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    tool = GoogleGeocoderTool()
    result = tool.geocode("Guadalajara")
    assert result is not None
    assert result["status"] == "OK"
    assert result["location"]["lat"] == 20.67
    assert result["location"]["lng"] == -103.35
    assert result["formatted_address"] == "Guadalajara, Jalisco, Mexico"


def test_google_geocoder_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    tool = GoogleGeocoderTool()
    with pytest.raises(RuntimeError):
        tool.geocode("Guadalajara")
