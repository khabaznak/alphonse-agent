"""Non-authoritative text-token estimates for evaluation telemetry.

These diagnostics never reserve, settle, enforce or price a model request.
They cover only provided text, not provider framing, hidden reasoning, native
tools, images or cached input. A word-based estimate is particularly weak for
languages without whitespace word boundaries. Calibrate against measured
usage for each model/language; an estimate is not an upper bound.
"""
from __future__ import annotations

import math
import re
from typing import Any


def usage_diagnostics(*, input_text: str | None, output_text: str | None,
                      provider_usage: dict[str, Any] | None = None,
                      bytes_per_token: float = 4.0, tokens_per_word: float = 1.33,
                      selected_method: str = 'utf8_bytes',
                      calibration: str | None = None) -> dict[str, Any]:
    """Return measured values where available, plus separate text estimates.

    Values are deliberately returned with provenance for each dimension.
    Never infer provider output usage from the input or turn missing text into
    a zero. Text content itself is not included in the returned telemetry.
    """
    if selected_method not in {'utf8_bytes', 'words'}:
        raise ValueError('invalid_estimate_method')
    for value in (bytes_per_token, tokens_per_word):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid_estimate_calibration')
    if calibration is not None and (not isinstance(calibration, str) or not calibration.strip()):
        raise ValueError('invalid_calibration_label')
    if any(value is not None and not isinstance(value, str) for value in (input_text, output_text)):
        raise ValueError('invalid_observed_text')
    source = provider_usage if isinstance(provider_usage, dict) else {}
    result = {}
    for direction, text in (('input', input_text), ('output', output_text)):
        reported = source.get(direction + '_tokens')
        measured = reported if type(reported) is int and reported >= 0 else None
        if text is None:
            estimates = {'utf8_bytes': None, 'words': None}
        else:
            estimates = {'utf8_bytes': math.ceil(len(text.encode('utf-8')) / bytes_per_token),
                         'words': math.ceil(len(re.findall(r'\S+', text)) * tokens_per_word)}
        selected = estimates[selected_method]
        result[direction] = {
            'tokens': measured if measured is not None else selected,
            'source': 'provider' if measured is not None else 'estimated' if selected is not None else 'unknown',
            'provider_tokens': measured,
            'text_estimates': estimates,
        }
    return {
        **result,
        'estimate_method': selected_method,
        'bytes_per_token': bytes_per_token,
        'tokens_per_word': tokens_per_word,
        'calibration': calibration or 'unverified_for_model_and_language',
        'scope': 'observed_text_only',
        'limitations': ['Not a provider charge or a safe upper bound.',
                        'Provider framing, hidden reasoning, images and tools may be absent.',
                        'Whitespace word counts can underestimate scripts without word spaces.'],
    }
