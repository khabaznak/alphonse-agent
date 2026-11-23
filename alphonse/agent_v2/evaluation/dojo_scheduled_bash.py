"""Real-model scheduled-file Dojo with fake Telegram and disposable host data.

Trusted host Bash has the local user's permissions; the fixture is not a sandbox.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from . import dojo_telegram_reminder as base


def run(model, emit):
    with tempfile.TemporaryDirectory(prefix='dojo-scheduled-reading-', dir='/private/tmp') as temp:
        root = Path(temp)
        source = root / 'inspection.txt'
        source.write_text('Inspection reading: 73.6 psi\n', encoding='utf-8')
        prompt = (f'In one minute, read the latest inspection reading in {source} '
                  'and tell me the reading and unit. Do not modify the record.')
        original_build = base.build_runtime_host

        def build(**kwargs):
            runtime = original_build(**kwargs)
            runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)
            return runtime

        with patch.object(base, 'PROMPT', prompt), patch.object(base, 'build_runtime_host', build):
            result = base.run(model, emit, timeout=240)
        responses = result.get('fake_telegram_sends', [])
        due_text = str(responses[1].get('text') or '') if len(responses) > 1 else ''
        result.update(case='telegram-scheduled-host-reading',
                      source_unchanged=source.read_text(encoding='utf-8') == 'Inspection reading: 73.6 psi\n',
                      due_response_mentions_reading='73.6' in due_text,
                      due_response_mentions_unit='psi' in due_text.casefold(),
                      response_semantics='requires_manual_review')
        result['limitations'].extend(['Trusted host Bash is unrestricted; only the fixture is disposable.',
                                     'Telegram is simulated; no real provider delivery was verified.'])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': 'telegram-scheduled-host-reading', 'model': args.model})
        try:
            result = run(args.model, emit)
        except Exception as exc:
            emit({'event': 'finished', 'status': 'runner_error', 'error_type': type(exc).__name__})
            raise
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
