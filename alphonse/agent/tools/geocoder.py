from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any


class GeocoderTool:
    def geocode(
        self,
        address: str,
        *,
        language: str | None = None,
        region: str | None = None,
    ) -> dict[str, Any] | None:
        raise NotImplementedError


class GoogleGeocoderTool(GeocoderTool):
    def __init__(self, api_key_env: str = "GOOGLE_MAPS_API_KEY") -> None:
        self._api_key_env = api_key_env

    def geocode(
        self,
        address: str,
        *,
        language: str | None = None,
        region: str | None = None,
    ) -> dict[str, Any] | None:
        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is not configured")
        address = str(address or "").strip()
        if not address:
            return None
        params = {
            "address": address,
            "key": api_key,
        }
        if language:
            params["language"] = language
        if region:
            params["region"] = region
        url = "https://maps.googleapis.com/maps/api/geocode/json?" + urllib.parse.urlencode(
            params
        )
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        status = str(data.get("status") or "")
        if status != "OK":
            return {
                "status": status,
                "error_message": data.get("error_message"),
                "results": [],
            }
        results = data.get("results") or []
        if not results:
            return {"status": status, "results": []}
        first = results[0]
        geometry = first.get("geometry") or {}
        location = geometry.get("location") or {}
        return {
            "status": status,
            "formatted_address": first.get("formatted_address"),
            "place_id": first.get("place_id"),
            "types": first.get("types") or [],
            "location": {
                "lat": location.get("lat"),
                "lng": location.get("lng"),
            },
            "raw": first,
        }
