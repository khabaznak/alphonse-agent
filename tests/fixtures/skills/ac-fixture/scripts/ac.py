"""Minimal disposable device CLI. No network, credentials, or household code."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=['temperature', 'set', 'verify'])
parser.add_argument('--room', required=True)
parser.add_argument('--target', type=float)
args = parser.parse_args()


def fail(code):
    print(json.dumps({'error': code}))
    raise SystemExit(2)


bindings = json.loads(os.environ['ALPHONSE_SKILL_BINDINGS'])
ids = bindings['settings']['configuration']['rooms'].get(args.room, [])
if not ids:
    fail('unknown_room')
if len(ids) != 1:
    fail('ambiguous_room')
path = Path(bindings['device']['path'])
state = json.loads(path.read_text())
device = state['devices'].get(ids[0])
if not device or not device['available']:
    fail('device_unavailable')
if args.operation in ('set', 'verify') and args.target is None:
    fail('target_required')
if args.operation == 'set':
    device['writes'] += 1
    if not device.get('ignore_write'):
        device['target'] = args.target
    path.write_text(json.dumps(state))
    if device.get('fail_after_write'):
        fail('acknowledgement_lost')
    print(json.dumps({'acknowledged': True, 'source': ids[0]}))
else:
    if args.operation == 'verify' and device['target'] != args.target:
        fail('state_mismatch')
    print(json.dumps({'room': args.room, 'source': ids[0],
                      'temperature': device['temperature'], 'unit': device['unit'],
                      'target': device['target'],
                      'observed_at': datetime.now(timezone.utc).isoformat(),
                      'verified': args.operation == 'verify'}))
