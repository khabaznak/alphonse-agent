"""Compare an explicit synthetic owner budget on the device-service Dojo.

This is a process-local test injection. It changes neither product defaults
nor existing checkpoints and is not an authorization or recovery mechanism.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
from unittest.mock import patch

from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_processor
from alphonse.agent_v2.core.inference import openai_codex
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits
from alphonse.agent_v2.evaluation import dojo_device_service
from alphonse.agent_v2.evaluation.owner_dojo_shopping import run


def provider_event_counts(raw):
    """Only allowlisted event types are retained; never keep provider content."""
    allowed = {'agent_message', 'command_execution', 'file_change', 'web_search',
               'mcp_tool_call', 'reasoning', 'error'}
    try:
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        counts = Counter()
        for event in events:
            if not isinstance(event, dict) or event.get('type') != 'item.completed':
                continue
            item = event.get('item')
            kind = item.get('type') if isinstance(item, dict) else None
            counts[kind if isinstance(kind, str) and kind in allowed else 'other'] += 1
        return dict(counts)
    except (TypeError, ValueError):
        return {'unavailable': 1}


def experiment_limits(original_total, final_reserve, elapsed_ms, final_reserve_ms, *, actions, model_calls):
    expected = {'actions': 5, 'model_calls': 8, 'input_tokens': 524288, 'output_tokens': 131072}
    if dict(original_total) != expected or type(actions) is not int or type(model_calls) is not int:
        raise ValueError('dojo_budget_baseline_changed')
    if not 5 <= actions <= 12 or not 8 <= model_calls <= 12:
        raise ValueError('dojo_budget_experiment_out_of_bounds')
    total = {**expected, 'actions': actions, 'model_calls': model_calls}
    return BudgetLimits(total, final_reserve, elapsed_ms, final_reserve_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--actions', type=int, required=True)
    parser.add_argument('--model-calls', type=int, required=True)
    parser.add_argument('--disable-provider-shell', action='store_true',
                        help='try CLI shell-tool feature flags; does not disable Alphonse Bash')
    args = parser.parse_args()
    # Validate before reserving a journal or contacting the model.
    trial = experiment_limits({'actions': 5, 'model_calls': 8, 'input_tokens': 524288, 'output_tokens': 131072},
                              {'actions': 0, 'model_calls': 1, 'input_tokens': 65536, 'output_tokens': 16384},
                              120000, 5000, actions=args.actions, model_calls=args.model_calls)
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': dojo_device_service.CASE_ID, 'goal': dojo_device_service.GOAL,
              'model': args.model, 'image_id': args.image_id, 'source_tree_sha256': digest.hexdigest(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'experiment_budget': trial.to_dict(),
              'provider_default_shell_flags': True,
              'provider_shell_flags_requested': args.disable_provider_shell,
              'scope': 'synthetic owner worker; process-local budget override; disposable container'})
        def limits(total, final, elapsed, final_ms):
            return experiment_limits(total, final, elapsed, final_ms,
                                     actions=args.actions, model_calls=args.model_calls)
        original_usage = openai_codex._reported_codex_usage
        def observe_usage(raw):
            usage = original_usage(raw)
            emit({'event': 'provider_event_summary', 'completed_item_types': provider_event_counts(raw),
                  'usage': usage})
            return usage
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(owner_read_processor, 'BudgetLimits', limits))
                stack.enter_context(patch.object(openai_codex, '_reported_codex_usage', observe_usage))
                if args.disable_provider_shell:
                    cli = shutil.which('codex')
                    if not cli:
                        raise ValueError('dojo_codex_cli_unavailable')
                    directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='dojo-provider-')))
                    wrapper = directory / 'codex-no-shell'
                    wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(cli)
                                       + ' --disable shell_tool --disable unified_exec "$@"\n', encoding='utf-8')
                    wrapper.chmod(0o700)
                    stack.enter_context(patch.object(openai_codex, '_resolve_cli', return_value=(str(wrapper), None)))
                result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                             model=args.model, timeout_seconds=120, scenario=dojo_device_service,
                             full_registry=True)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'task_passed': None,
                  'error_type': type(exc).__name__})
            raise
        result['experiment_budget'] = trial.to_dict()
        result['provider_default_shell_flags'] = True
        result['provider_shell_flags_requested'] = args.disable_provider_shell
        result['limitations'].append('One process-local budget experiment; product defaults and authorization are unchanged.')
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
