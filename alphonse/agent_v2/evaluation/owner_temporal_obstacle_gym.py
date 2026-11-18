"""Offline lifecycle obstacles; synthetic effects do not establish capability."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .owner_obstacle_gym import FixtureInterrupted, FixtureUnavailable, FixtureWorld, _effects_consistent, grade, load_corpus


def _local_time(value: Any, zone_name: Any) -> tuple[str, dict[int, str]]:
    """Classify a wall time by UTC round-trip; folds are never inferred."""
    try:
        wall = datetime.fromisoformat(value)
        if wall.tzinfo is not None or not isinstance(zone_name, str):
            return 'invalid', {}
        zone = ZoneInfo(zone_name)
        choices = {}
        for fold in (0, 1):
            utc = wall.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
            back = utc.astimezone(zone)
            if back.replace(tzinfo=None) == wall and back.fold == fold:
                choices[fold] = utc.isoformat()
        return ('nonexistent' if not choices else 'ambiguous' if len(choices) == 2 else 'resolved'), choices
    except (TypeError, ValueError, ZoneInfoNotFoundError, OverflowError):
        return 'invalid', {}


class TemporalFixtureWorld(FixtureWorld):
    """Synthetic state with explicit version and event-id checks; not a sandbox."""
    def __init__(self, fixture):
        super().__init__(fixture)
        self.lifecycle = deepcopy(fixture['lifecycle'])
        self._seen_events: set[tuple[str, str]] = set()

    def _record(self, kind, arguments):
        previous = next((effect for effect in self.effects if effect.get('request_id') == arguments['request_id']), None)
        intended = {'kind': kind, **deepcopy(arguments)}
        if previous is not None:
            return 'duplicate' if previous == intended else 'conflict'
        self.effects.append(intended)
        return 'saved'

    def act(self, actor, operation, arguments):
        operations = {'change_reminder', 'cancel_reminder', 'change_schedule', 'cancel_schedule', 'change_subscription', 'cancel_subscription', 'receive_event', 'fire_occurrence', 'resolve_local_time'}
        if operation not in operations:
            return super().act(actor, operation, arguments)
        if not isinstance(arguments, dict):
            raise ValueError('invalid_fixture_action')
        result = {}
        if actor != self.actor or operation in self._denied:
            status = 'denied'
        elif operation not in self._allowed:
            status = 'unavailable'
        elif operation in {'change_reminder', 'cancel_reminder', 'change_schedule', 'cancel_schedule', 'change_subscription', 'cancel_subscription'}:
            status = self._change(operation, arguments)
        elif operation == 'receive_event':
            status = self._receive_event(arguments)
        elif operation == 'fire_occurrence':
            status = self._fire_occurrence(arguments)
        else:
            status, choices = _local_time(arguments.get('local_time'), arguments.get('timezone')) if set(arguments) == {'local_time', 'timezone'} else ('invalid', {})
            result['choices_utc'] = choices
        self.events.append({'actor': actor, 'operation': operation, 'status': status})
        return {'status': status, **result}

    def _change(self, operation, args):
        specs = {
            'change_reminder': ('reminders', 'reminder_id', {'request_id', 'reminder_id', 'expected_version', 'due_at'}),
            'cancel_reminder': ('reminders', 'reminder_id', {'request_id', 'reminder_id', 'expected_version'}),
            'change_schedule': ('schedules', 'schedule_id', {'request_id', 'schedule_id', 'expected_version', 'local_time', 'timezone', 'fold'}),
            'cancel_schedule': ('schedules', 'schedule_id', {'request_id', 'schedule_id', 'expected_version'}),
            'change_subscription': ('subscriptions', 'subscription_id', {'request_id', 'subscription_id', 'expected_version', 'source', 'event', 'action'}),
            'cancel_subscription': ('subscriptions', 'subscription_id', {'request_id', 'subscription_id', 'expected_version'}),
        }
        collection, id_field, fields = specs[operation]
        if (set(args) != fields or not isinstance(args.get('request_id'), str) or not args['request_id']
                or not isinstance(args.get(id_field), str) or not args[id_field]
                or type(args.get('expected_version')) is not int or args['expected_version'] < 1):
            return 'invalid'
        previous = next((effect for effect in self.effects if effect.get('request_id') == args['request_id']), None)
        intended = {'kind': operation, **deepcopy(args)}
        if previous is not None:
            return 'duplicate' if previous == intended else 'conflict'
        item = self.lifecycle.get(collection, {}).get(args[id_field])
        if not isinstance(item, dict) or item.get('owner') != self.actor:
            return 'denied'
        if item.get('status') != 'active' or type(item.get('version')) is not int or item['version'] != args['expected_version']:
            return 'stale'
        if operation == 'change_reminder':
            try:
                due = datetime.fromisoformat(args['due_at'])
                if due.tzinfo is None or due.utcoffset() is None:
                    return 'invalid'
            except (TypeError, ValueError):
                return 'invalid'
        if operation == 'change_subscription' and any(not isinstance(args[key], str) or not args[key] for key in ('source', 'event', 'action')):
            return 'invalid'
        if operation == 'change_schedule':
            classification, choices = _local_time(args['local_time'], args['timezone'])
            fold = args['fold']
            if classification == 'nonexistent' or (classification == 'ambiguous' and fold is None):
                return 'clarification'
            if classification not in {'resolved', 'ambiguous'} or type(fold) is not int or fold not in choices:
                return 'invalid'
        self.effects.append(intended)
        if operation == 'change_reminder':
            item['due_at'] = args['due_at']
        elif operation in {'cancel_reminder', 'cancel_schedule', 'cancel_subscription'}:
            item['status'] = 'cancelled'
        elif operation == 'change_subscription':
            item.update(source=args['source'], event=args['event'], action=args['action'])
        else:
            item.update(local_time=args['local_time'], timezone=args['timezone'], fold=args['fold'], next_utc=choices[args['fold']])
        item['version'] += 1
        return 'saved'

    def _receive_event(self, args):
        if (set(args) != {'subscription_id', 'event_id', 'source', 'event'}
                or any(not isinstance(value, str) or not value for value in args.values())):
            return 'invalid'
        item = self.lifecycle.get('subscriptions', {}).get(args['subscription_id'])
        if not isinstance(item, dict) or item.get('owner') != self.actor:
            return 'denied'
        if item.get('status') != 'active':
            return 'suppressed'
        if item.get('source') != args['source'] or item.get('event') != args['event']:
            return 'unmatched'
        key = (args['subscription_id'], args['event_id'])
        if key in self._seen_events:
            return 'duplicate'
        self._seen_events.add(key)
        self.effects.append({'kind': 'event_action', 'subscription_id': args['subscription_id'],
                             'event_id': args['event_id'], 'action': item['action']})
        return 'saved'

    def _fire_occurrence(self, args):
        if (set(args) != {'schedule_id', 'occurrence_id'} or
                any(not isinstance(value, str) or not value for value in args.values())):
            return 'invalid'
        item = self.lifecycle.get('schedules', {}).get(args['schedule_id'])
        if not isinstance(item, dict) or item.get('owner') != self.actor:
            return 'denied'
        if item.get('status') != 'active':
            return 'suppressed'
        key = ('schedule:' + args['schedule_id'], args['occurrence_id'])
        if key in self._seen_events:
            return 'duplicate'
        self._seen_events.add(key)
        self.effects.append({'kind': 'schedule_occurrence', 'schedule_id': args['schedule_id'],
                             'occurrence_id': args['occurrence_id'], 'action': item['action']})
        return 'saved'


def run_temporal_corpus(path: str | Path, adapter=None) -> dict:
    """Run trusted offline development fixtures with independent expectations."""
    corpus, digest = load_corpus(path)
    if adapter is not None and getattr(adapter, 'offline', False) is not True:
        raise ValueError('obstacle_adapter_must_declare_offline')
    rows = []
    for case in corpus['cases']:
        if adapter is None:
            rows.append({'id': case['id'], 'availability': 'unavailable', 'passed': None, 'reason': 'adapter_not_configured'})
            continue
        world = TemporalFixtureWorld(case['fixture'])
        try:
            result = adapter.run(deepcopy(case['input']), world)
            if not isinstance(result, dict) or result.get('status') not in {'completed', 'blocked', 'unavailable', 'clarification'}:
                raise ValueError('invalid_adapter_result')
            evaluation = grade(case, result, world)
            evaluation['checks']['final_state'] = world.lifecycle == case['oracle']['expected_state']
            evaluation['boundary_passed'] = all(evaluation['checks'].values())
            evaluation['passed'] = False if not evaluation['boundary_passed'] else (True if result['status'] == evaluation['target_status'] and evaluation['semantic_assessment'] != 'unassessed' else None)
            rows.append({'id': case['id'], 'availability': 'observed', **evaluation, 'status': result['status'],
                         'events': deepcopy(world.events), 'effects': deepcopy(world.effects)})
        except Exception as exc:
            availability = 'interrupted' if isinstance(exc, FixtureInterrupted) else 'unavailable' if isinstance(exc, FixtureUnavailable) else 'error'
            consistent = _effects_consistent(world.effects, case['oracle']['expected_effects'], complete=False)
            rows.append({'id': case['id'], 'availability': availability, 'passed': None if consistent else False,
                         'observed_effects_consistent': consistent, 'events': deepcopy(world.events), 'effects': deepcopy(world.effects)})
    return {'schema_version': 1, 'corpus_sha256': digest, 'simulation': True, 'cases': rows,
            'limitations': ['Trusted adapters and mutable synthetic state are not an adversarial oracle or a network sandbox.',
                'Fixture effects and event delivery do not demonstrate real product behavior, autonomous reasoning or external effects.',
                'Only deliberately raised fixture interruptions preserve in-memory evidence; process termination may lose the report.']}
