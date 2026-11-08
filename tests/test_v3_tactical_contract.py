from copy import deepcopy

import pytest

from alphonse.agent_v2.core.intelligence.v3.tactical_contract import valid_tactical_response

QUESTION = {"question_id": "q", "type": "noul", "instructions": "Was it observed?",
            "criteria": {"true": "Observed", "false": "Not observed"}}
ACTION = {"tool_id": "native.read", "arguments": {"id": "fixture"}, "acceptance_questions": [QUESTION]}


def test_valid_action_and_sequences():
    assert valid_tactical_response(ACTION)
    assert valid_tactical_response({"actions": [ACTION, ACTION]})
    assert valid_tactical_response({"browser_fill_sequence": [ACTION, ACTION, ACTION]})


@pytest.mark.parametrize("value", [None, [], {"action": ACTION}, {"arguments": ACTION["arguments"]},
    {"tool_id": "native.read", "arguments": {}},
    {"actions": [ACTION]}, {"actions": [ACTION, {"tool_id": "native.read"}]},
    {**ACTION, "extra": "fixture"}, {**ACTION, "actions": [ACTION, ACTION]},
    {**ACTION, "acceptance_questions": []},
    {**ACTION, "acceptance_questions": [{**QUESTION, "criteria": {"true": "yes"}}]}])
def test_invalid_outer_contract_is_rejected(value):
    assert not valid_tactical_response(value)


def _wire(arguments='{"field":"fixture"}'):
    return {'action': {'tool_id': 'native.browser_workflow_fill', 'arguments_json': arguments,
                       'acceptance_questions': [QUESTION]},
            'actions': [], 'browser_fill_sequence': [], 'uncertainty_assessment': None}


def test_wire_schema_is_strict_and_decodes_arguments():
    from alphonse.agent_v2.core.inference.tactical_wire import SCHEMA, decode
    def walk(schema):
        if isinstance(schema, dict):
            if schema.get('type') == 'object':
                assert schema.get('additionalProperties') is False
                assert set(schema.get('required', [])) == set(schema.get('properties', {}))
            for value in schema.values():
                walk(value)
        elif isinstance(schema, list):
            for value in schema:
                walk(value)
    walk(SCHEMA)
    assert decode(_wire())['arguments'] == {'field': 'fixture'}


@pytest.mark.parametrize('change,code', [
    (lambda x: x.update(action=None), 'openai_codex_invalid_tactical_schema'),
    (lambda x: x.update(actions=[x['action']]), 'openai_codex_invalid_tactical_schema'),
    (lambda x: x['action'].update(arguments_json='{"secret":'), 'openai_codex_invalid_json'),
    (lambda x: x['action'].update(arguments_json='[1]'), 'openai_codex_invalid_tactical_schema'),
    (lambda x: x['action'].pop('tool_id'), 'openai_codex_invalid_tactical_schema'),
    (lambda x: x.update(extra='synthetic-private-output'), 'openai_codex_invalid_tactical_schema'),
])
def test_wire_rejections_are_categorized_without_values(change, code):
    from alphonse.agent_v2.core.inference.tactical_wire import decode
    value = _wire()
    change(value)
    with pytest.raises(ValueError) as caught:
        decode(value)
    assert str(caught.value) == code


def test_frozen_corpus_reports_unsafe_acceptance_without_fixture_content():
    import json
    from alphonse.agent_v2.evaluation.tactical_wire_corpus import report
    value = report(1)
    assert value['arms']['legacy_parser']['unsafe_accepts'] == 1
    assert value['arms']['structured_wire']['wrong'] == 0
    assert 'fixture' not in json.dumps(value).lower()


def test_transport_error_does_not_expose_provider_text():
    from alphonse.agent_v2.core.inference.models import InferencePurpose, InferenceRequest
    from alphonse.agent_v2.core.intelligence.v3.executor import _safe_tactical_generate
    class Provider:
        def generate_json(self, request):
            raise ConnectionError('synthetic-private-output')
    with pytest.raises(RuntimeError) as caught:
        _safe_tactical_generate(Provider(), InferenceRequest(prompt='', purpose=InferencePurpose.TACTICAL_ACTION))
    assert str(caught.value) == 'tactical_provider_error'
    assert caught.value.__suppress_context__
