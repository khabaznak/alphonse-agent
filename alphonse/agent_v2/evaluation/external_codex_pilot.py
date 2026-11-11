"""Prepare and inspect a synthetic Codex reference attempt, without dispatching it.

Neither argument construction nor JSONL inspection supplies isolation,
authorization, an external effect oracle or a spending cap. They do not invoke
Codex or access credentials. Only synthetic traces belong in this workflow.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def response_schema(case: str) -> dict[str, Any]:
    """Return a public output contract, not a source record or answer key."""
    string = {'type': 'string'}
    if case == 'clock':
        properties = {key: string.copy() for key in ('local_time', 'local_date', 'timezone')}
    elif case == 'prescription':
        properties = {'status': string.copy(), 'medications': {'type': 'array', 'items': {
            'type': 'object', 'properties': {key: string.copy() for key in ('name', 'dose', 'time')},
            'required': ['name', 'dose', 'time'], 'additionalProperties': False}}}
    elif case == 'traffic':
        variants = [
            {'type': 'object', 'properties': {'status': {'const': 'verified'}, 'subject': string.copy(),
                'ticket_ids': {'type': 'array', 'items': string.copy()}},
             'required': ['status', 'subject', 'ticket_ids'], 'additionalProperties': False},
            {'type': 'object', 'properties': {'status': {'const': 'blocked'}, 'reason': string.copy()},
             'required': ['status', 'reason'], 'additionalProperties': False},
        ]
        # Structured Outputs requires an object at the root; the verified and
        # blocked variants are wrapped so an evaluator can unwrap them safely.
        return {'type': 'object', 'properties': {'result': {'anyOf': variants}},
                'required': ['result'], 'additionalProperties': False}
    else:
        raise ValueError('unsupported_reference_case')
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def reference_response(case: str, final_text: str) -> dict[str, Any] | None:
    """Parse and schema-check private output for a separate fixture oracle.

    Return value may contain synthetic case data or bearer material: keep it in
    evaluator-controlled storage and never include it in a sanitized summary.
    A schema-conforming response is not an independently verified result.
    """
    from jsonschema import Draft202012Validator
    schema = response_schema(case)
    try:
        value = json.loads(final_text)
    except (TypeError, ValueError):
        return None
    if not Draft202012Validator(schema).is_valid(value):
        return None
    return value['result'] if case == 'traffic' else value


def build_argv(*, executable: Path, workspace: Path, model: str,
               output_last_message: Path, output_schema: Path) -> list[str]:
    """Build fixed noninteractive arguments; the caller must not execute yet.

    Supply the synthetic prompt through stdin, never argv. Separate enforcement
    must establish an isolated environment, approved provider and spending scope.
    Paths and model here are only syntactically checked, not attested.
    """
    paths = (executable, workspace, output_last_message, output_schema)
    if any(not isinstance(p, Path) or not p.is_absolute() or '\x00' in str(p) for p in paths):
        raise ValueError('absolute_paths_required')
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', model):
        raise ValueError('invalid_model')
    return [str(executable), 'exec', '--json', '--ephemeral', '--ignore-user-config',
            '--sandbox', 'read-only', '--skip-git-repo-check', '--model', model,
            '--cd', str(workspace), '--output-schema', str(output_schema),
            '--output-last-message', str(output_last_message), '-']


def inspect_trace(jsonl: str, *, final_text: str | None, exit_code: int | None) -> dict[str, Any]:
    """Summarize a single complete Codex JSONL turn without exposing raw trace.

    Observed command attempts and token counters do not prove an action, cost,
    result, delivery, or which process contacted a fixture. A separate oracle
    must evaluate a final structured response against fixture audit. Unknown or
    interrupted traces cannot become success merely because they contain text.
    """
    if not isinstance(jsonl, str) or (final_text is not None and not isinstance(final_text, str)):
        raise ValueError('invalid_trace')
    if exit_code is not None and type(exit_code) is not int:
        raise ValueError('invalid_exit_code')
    records = []
    malformed = False
    for line in jsonl.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            malformed = True
            break
        if not isinstance(event, dict):
            malformed = True
            break
        records.append(event)
    completions = [event for event in records if event.get('type') == 'turn.completed']
    failures = [event for event in records if event.get('type') in {'turn.failed', 'error'}]
    completed = bool(exit_code == 0 and not malformed and len(completions) == 1 and not failures
                     and records and records[-1].get('type') == 'turn.completed')
    usage = completions[0].get('usage') if len(completions) == 1 else None
    usage = usage if isinstance(usage, dict) else {}
    def nonnegative(value):
        return value if type(value) is int and value >= 0 else None
    reported_usage = {name: nonnegative(usage.get(name)) for name in ('input_tokens', 'output_tokens')}
    commands = [event.get('item') for event in records if event.get('type') == 'item.completed'
                and isinstance(event.get('item'), dict) and event['item'].get('type') == 'command_execution']
    command_count = len(commands)
    answer = None
    if completed and final_text:
        try:
            value = json.loads(final_text)
        except (TypeError, ValueError):
            value = None
        if isinstance(value, dict):
            answer = value
    return {'status': 'completed' if completed else 'incomplete_or_failed',
            'structured_response_available': answer is not None,
            'response_source': 'last_message_file' if answer is not None else 'unknown',
            'command_attempts': command_count, 'reported_usage': reported_usage,
            'usage_source': 'codex_cli_reported' if all(v is not None for v in reported_usage.values()) else 'partial_or_unknown',
            'model_identity': 'unverified', 'delivery': 'unassessed',
            'first_useful_response_ms': None, 'fixture_effect': 'unassessed',
            'decision_qualified': False,
            'limitations': ['Raw trace, commands and answer omitted; independently retain approved sanitized evidence.',
                            'JSON parsing is not schema validation; only one completed turn is accepted.',
                            'Time, model identity, cost, delivery and agent-to-fixture attribution are unassessed.']}
