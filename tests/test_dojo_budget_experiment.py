import pytest

from alphonse.agent_v2.evaluation.dojo_budget_experiment import experiment_limits, provider_event_counts


def test_experiment_changes_only_declared_dimensions_and_rejects_new_baseline():
    base = {'actions': 5, 'model_calls': 8, 'input_tokens': 524288, 'output_tokens': 131072}
    final = {'actions': 0, 'model_calls': 1, 'input_tokens': 65536, 'output_tokens': 16384}
    limits = experiment_limits(base, final, 120000, 5000, actions=8, model_calls=10)
    assert limits.total == {**base, 'actions': 8, 'model_calls': 10}
    assert limits.final_reserve == final
    assert limits.elapsed_ms == 120000
    with pytest.raises(ValueError):
        experiment_limits({**base, 'actions': 6}, final, 120000, 5000, actions=8, model_calls=10)
    with pytest.raises(ValueError):
        experiment_limits(base, final, 120000, 5000, actions=13, model_calls=10)


def test_provider_event_counts_never_retains_text_or_unknown_item_type():
    raw = ('{"type":"item.completed","item":{"type":"agent_message","text":"private"}}\n'
           '{"type":"item.completed","item":{"type":"private type","text":"private"}}\n')
    result = provider_event_counts(raw)
    assert result == {'agent_message': 1, 'other': 1}
    assert 'private' not in str(result)
    assert provider_event_counts('{broken') == {'unavailable': 1}
