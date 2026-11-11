"""Fixed-endpoint, opt-in HTTP transport for a synthetic Jev pilot.

Not an authorization or isolation boundary. The caller must separately
obtain approval for synthetic data, provider, retention, attempts and budget,
and must pass this transport through the attempt meter. A network error can
occur after the provider processed a request; never retry without metering.
No credentials are read from the environment or written to the result.
"""
from __future__ import annotations

import json
from io import BytesIO
import math
from typing import Any

from alphonse.agent_v2.evaluation.jev_pilot_payload import serialize_payload
from alphonse.agent_v2.system_one import DEFAULT_SYSTEM_ONE_URL, SystemOneRequestError, _provider_request_error_code


class JevPilotHTTPTransport:
    """One request per call, no redirects, ambient proxy or HTTP retries.

    ``enabled`` only enables this local code path; it does not establish user
    approval, a hard spending cap, synthetic-data isolation or endpoint trust.
    """

    def __init__(self, *, enabled: bool = False, max_response_bytes: int = 1_048_576):
        if type(enabled) is not bool or type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError('jev_pilot_invalid_http_config')
        self.enabled, self.max_response_bytes = enabled, max_response_bytes

    def __call__(self, url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if not self.enabled:
            raise ValueError('jev_pilot_http_disabled')
        if url != DEFAULT_SYSTEM_ONE_URL:
            raise ValueError('jev_pilot_endpoint_denied')
        if (not isinstance(api_key, str) or not api_key.strip() or not isinstance(payload, dict)
                or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('jev_pilot_http_invalid_request')
        import requests
        data = serialize_payload(payload)
        failure, status, value, provider_code = None, None, None, ''
        try:
            with requests.Session() as session:
                session.trust_env = False
                session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
                with session.post(url, data=data, headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
                                  allow_redirects=False, timeout=(min(10, timeout), timeout), stream=True) as response:
                    status = response.status_code
                    if status == 200:
                        chunks, size = [], 0
                        for chunk in response.iter_content(chunk_size=8192):
                            size += len(chunk)
                            if size > self.max_response_bytes:
                                raise ValueError('jev_pilot_response_too_large')
                            chunks.append(chunk)
                        value = json.loads(b''.join(chunks))
                    elif status in (400, 413, 415, 422):
                        # Preserve only recognized machine codes; provider
                        # prose and credential-bearing errors are discarded.
                        try:
                            error_body = bytearray()
                            for chunk in response.iter_content(chunk_size=8192):
                                error_body.extend(chunk[:max(0, 32768 - len(error_body))])
                                if len(error_body) >= 32768:
                                    break
                            provider_code = _provider_request_error_code(BytesIO(error_body))
                        except Exception:
                            provider_code = ''
        except requests.RequestException:
            # Exceptions can hold Authorization and response previews; raise
            # after leaving the handler, with no exception chain or body text.
            failure = 'connection'
        except Exception:
            failure = 'invalid'
        if failure == 'connection':
            raise RuntimeError('system_one_connection_failed')
        if failure:
            raise ValueError('system_one_response_invalid')
        if status in (400, 413, 415, 422):
            raise SystemOneRequestError(status, provider_code)
        if status == 401:
            raise ValueError('system_one_api_key_invalid')
        if status in (429, 529):
            raise RuntimeError(f'system_one_temporarily_unavailable:{status}')
        if status != 200:
            raise RuntimeError(f'system_one_http_error:{status if type(status) is int and 100 <= status <= 999 else 500}')
        if not isinstance(value, dict):
            raise ValueError('system_one_response_invalid')
        return value
