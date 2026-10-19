import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.evaluation import household_live


def provider(monkeypatch, tmp_path, events):
    monkeypatch.setattr(household_live, "configured_model", lambda: "fixture-model")
    monkeypatch.setattr(household_live.Path, "home", lambda: tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(item) for item in events), stderr="")

    monkeypatch.setattr(household_live.subprocess, "run", run)
    return household_live.HouseholdLiveProvider(), commands


def test_runtime_json_is_not_double_encoded_and_transport_tools_are_disabled(monkeypatch, tmp_path):
    value = {"tool_id": "native.respond", "arguments": {"message": 'A quoted "result"\nwith another line.'}}
    instance, commands = provider(monkeypatch, tmp_path, [
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(value)}},
        {"type": "turn.completed", "usage": {"input_tokens": 42, "output_tokens": 12}},
    ])
    result = instance.generate_json(InferenceRequest(prompt="Compose a fixture action", purpose=InferencePurpose.TACTICAL_ACTION))
    assert result.json_value == value
    assert result.usage == {"input_tokens": 42, "output_tokens": 12}
    assert "--ignore-user-config" in commands[0]
    assert "features.shell_tool=false" in commands[0]
    assert "features.apps=false" in commands[0]
    assert "features.plugins=false" in commands[0]


def test_malformed_output_remains_a_failed_attempt_with_raw_fixture_evidence(monkeypatch, tmp_path):
    instance, _ = provider(monkeypatch, tmp_path, [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "malformed fixture output"}},
    ])
    with pytest.raises(RuntimeError, match="invalid_json"):
        instance.generate_json(InferenceRequest(prompt="Fixture", purpose=InferencePurpose.TACTICAL_ACTION))
    assert instance.calls[0]["status"] == "failed"
    assert instance.calls[0]["raw_output"] == "malformed fixture output"


def test_unexpected_transport_tool_activity_cannot_count_as_success(monkeypatch, tmp_path):
    instance, _ = provider(monkeypatch, tmp_path, [
        {"type": "item.completed", "item": {"type": "command_execution", "command": "fixture forbidden command"}},
    ])
    with pytest.raises(RuntimeError, match="unexpected_tool"):
        instance.generate_json(InferenceRequest(prompt="Fixture", purpose=InferencePurpose.TACTICAL_ACTION))
    assert instance.calls[0]["status"] == "failed"


def test_private_fixture_sentinel_is_rejected_before_any_provider_request(monkeypatch, tmp_path):
    instance, commands = provider(monkeypatch, tmp_path, [])
    with pytest.raises(RuntimeError, match="private_source_before_provider"):
        instance.generate_json(InferenceRequest(prompt="PRIVATE_SENTINEL", purpose=InferencePurpose.TACTICAL_ACTION))
    assert commands == []
    assert instance.calls[0]["status"] == "failed"


def test_simulated_clock_reaches_provider_system_instructions(monkeypatch, tmp_path):
    instance, _ = provider(monkeypatch, tmp_path, [
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{"content":"Synthetic response"}'}},
        {'type': 'turn.completed', 'usage': {'input_tokens': 42, 'output_tokens': 12}},
    ])
    instance.configure_fixture_clock('2026-02-11T08:00:00-06:00', 'America/Mexico_City')
    run = household_live.subprocess.run
    seen = []
    def inspect(command, **kwargs):
        from pathlib import Path
        seen.append((Path(kwargs['cwd']) / 'instructions.txt').read_text())
        return run(command, **kwargs)
    monkeypatch.setattr(household_live.subprocess, 'run', inspect)
    instance.generate_markdown(InferenceRequest(prompt='Explain a completed or failed task.', purpose=InferencePurpose.FINAL_RESPONSE))
    assert '2026-02-11T08:00:00-06:00' in seen[0]
    assert 'America/Mexico_City' in seen[0]
    assert instance.calls[0]['fixture_clock']['local_time'] == '2026-02-11T08:00:00-06:00'
