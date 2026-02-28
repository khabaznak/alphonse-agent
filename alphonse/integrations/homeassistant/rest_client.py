from __future__ import annotations

import time
from typing import Any

import requests

from alphonse.integrations.homeassistant.config import HomeAssistantConfig


class HomeAssistantTransportError(RuntimeError):
    pass


class HomeAssistantAuthError(HomeAssistantTransportError):
    pass


class HomeAssistantRateLimitError(HomeAssistantTransportError):
    pass


class HomeAssistantRestClient:
    def __init__(self, config: HomeAssistantConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "Content-Type": "application/json",
            }
        )

    def get_states(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/api/states")
        data = response.json()
        return data if isinstance(data, list) else []

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        response = self._request("GET", f"/api/states/{entity_id}", allow_not_found=True)
        if response is None:
            return None
        data = response.json()
        return data if isinstance(data, dict) else None

    def call_service(
        self,
        domain: str,
        service: str,
        data: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {}
        if isinstance(data, dict):
            payload.update(data)
        if isinstance(target, dict) and target:
            payload["target"] = target
        response = self._request("POST", f"/api/services/{domain}/{service}", json=payload)
        body = response.json()
        return body if isinstance(body, list) else []

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> requests.Response | None:
        url = f"{self._config.base_url}{path}"
        attempts = max(1, self._config.retry.max_attempts)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.request(
                    method,
                    url,
                    json=json,
                    timeout=self._config.request_timeout_sec,
                )
                if allow_not_found and response.status_code == 404:
                    return None
                if response.status_code in {401, 403}:
                    raise HomeAssistantAuthError(f"Home Assistant auth failed status={response.status_code}")
                if response.status_code == 429:
                    if attempt < attempts:
                        self._sleep_before_retry(attempt)
                        continue
                    raise HomeAssistantRateLimitError("Home Assistant rate limited")
                if 500 <= response.status_code < 600:
                    if attempt < attempts:
                        self._sleep_before_retry(attempt)
                        continue
                    raise HomeAssistantTransportError(
                        f"Home Assistant server error status={response.status_code}"
                    )
                if response.status_code >= 400:
                    raise HomeAssistantTransportError(
                        f"Home Assistant request failed status={response.status_code} body={response.text[:200]}"
                    )
                return response
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                self._sleep_before_retry(attempt)
            except (HomeAssistantAuthError, HomeAssistantRateLimitError, HomeAssistantTransportError):
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                self._sleep_before_retry(attempt)
        raise HomeAssistantTransportError(f"Home Assistant request failed: {last_error}")

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(
            self._config.retry.max_delay_sec,
            self._config.retry.base_delay_sec * (2 ** max(0, attempt - 1)),
        )
        time.sleep(max(0.0, delay))
