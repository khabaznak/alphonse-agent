"""Estimated usage is observable telemetry, never a billing or budget input."""
import math

import pytest

from alphonse.agent_v2.evaluation.token_usage_diagnostics import usage_diagnostics


def test_missing_provider_usage_is_estimated_separately_for_spanish_and_cjk():
    prompt, answer = 'Hola, ¿cuántos correos tengo?', '你好，世界'
    item = usage_diagnostics(input_text=prompt, output_text=answer)
    assert item['input']['source'] == item['output']['source'] == 'estimated'
    assert item['input']['tokens'] == math.ceil(len(prompt.encode('utf-8')) / 4)
    assert item['output']['tokens'] == math.ceil(len(answer.encode('utf-8')) / 4)
    assert item['calibration'] == 'unverified_for_model_and_language'
    assert prompt not in str(item) and answer not in str(item)


def test_measured_input_and_estimated_output_remain_distinct():
    item = usage_diagnostics(input_text='one two', output_text='tres cuatro',
        provider_usage={'input_tokens': 0}, selected_method='words',
        tokens_per_word=1.5, calibration='synthetic-model-es-v1')
    assert item['input']['tokens'] == 0 and item['input']['source'] == 'provider'
    assert item['output']['tokens'] == 3 and item['output']['source'] == 'estimated'
    assert item['calibration'] == 'synthetic-model-es-v1'


def test_unknown_text_and_invalid_provider_usage_are_not_zero():
    item = usage_diagnostics(input_text=None, output_text=None,
                             provider_usage={'input_tokens': True, 'output_tokens': -1})
    assert item['input']['source'] == item['output']['source'] == 'unknown'
    assert item['input']['tokens'] is None and item['output']['tokens'] is None


@pytest.mark.parametrize('bad', [0, -1, float('inf'), float('nan'), True])
def test_invalid_factor_rejected(bad):
    with pytest.raises(ValueError, match='calibration'):
        usage_diagnostics(input_text='a', output_text=None, bytes_per_token=bad)
