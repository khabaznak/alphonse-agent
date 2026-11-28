"""Bounded Agenda projection for scheduled work and the owner's Google Calendar."""
import os
import re
import sqlite3
from itertools import islice
from dateutil.rrule import rrulestr
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from alphonse.agent_v2.core.scheduled_tasks import compute_next_run_at, _parse_datetime
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped


def _calendar_owner(runtime, actor):
    owner = runtime.user_store.admin_user()
    return owner is not None and owner.user_id == actor and owner.is_active and owner.role in ('owner', 'admin')


def _calendar_source_status(runtime, actor):
    """Report local owner configuration without fetching tokens or events."""
    if not _calendar_owner(runtime, actor):
        return None
    service = getattr(runtime, "owner_calendar_service", None)
    if service is None:
        return "unavailable"
    try:
        binding, failure = service.binding(actor)
        if binding is not None:
            return "connected"
        if failure == "calendar_choice_required":
            return "selection_required"
        reconnect_required = False
        for record in service.accounts.integrations.list(provider_key="google"):
            connection = record.connection
            if connection.get("owner_user_id") == actor and "calendar_read" in connection.get("capabilities", ()):
                if (record.enabled and connection.get("status") == "connected"
                        and connection.get("health_code") == "healthy" and record.credential_ref
                        and SCOPES["google"]["calendar_read"] in connection.get("granted_scopes", ())
                        and not record.revocation_pending and not record.credential_deletion_pending):
                    return "grant_required"
                if connection.get("status") != "disconnected":
                    reconnect_required = True
        return "reconnect_required" if reconnect_required else "not_connected"
    except (sqlite3.Error, OSError, KeyError, ValueError, TypeError):
        return "unavailable"


def _calendar_events(runtime, actor, first, end, zone):
    service = runtime.owner_calendar_service

    def guard():
        if not _calendar_owner(runtime, actor):
            raise CheckStopped('blocked', 'agenda_calendar_owner_changed')

    try:
        if (not os.path.samefile(runtime.user_store.db_path, service.accounts.integrations.db_path)
                or not os.path.samefile(runtime.user_store.db_path, service.accounts.users.db_path)):
            return {'calendar_source_status': 'unavailable', 'calendar_event_status': 'unavailable'}
        observation = service.observe_agenda(actor, first, end, zone=zone, guard=guard)
        guard()
        if observation['status'] in ('complete', 'partial') and observation['authority'] == service.binding(actor)[0]:
            events = [{'id': f'google-calendar:{index}', 'title': event['title'], 'start': event['start'],
                       'end': event['end'], 'all_day': event['all_day'], 'end_unspecified': event['end_unspecified']}
                      for index, event in enumerate(observation['events'])]
            return {'calendar_source_status': 'connected', 'calendar_event_status': observation['status'],
                    'calendar_events': events}
        reason = observation.get('reason')
        if reason == 'reconnect_required':
            return {'calendar_source_status': 'reconnect_required', 'calendar_event_status': 'unavailable'}
        if reason in ('api_disabled', 'approval_required', 'credential_busy'):
            return {'calendar_source_status': 'connected', 'calendar_event_status': reason}
    except (CheckStopped, sqlite3.Error, OSError, ValueError, TypeError, KeyError):
        pass
    status = _calendar_source_status(runtime, actor)
    return {'calendar_source_status': status, 'calendar_event_status': 'unavailable'} if status is not None else {}


def agenda(runtime, *, actor, start_day, end_day, include_owner_calendar=False, include_calendar_events=False):
    member = runtime.user_store.get_user(actor)
    if member is None or not member.is_active:
        raise PermissionError("agenda_actor_required")
    first, end = date.fromisoformat(start_day), date.fromisoformat(end_day)
    if not 0 < (end - first).days <= 42:
        raise ValueError("agenda_range_must_be_1_to_42_days")
    zone = ZoneInfo(runtime.user_store.timezone(actor))
    start_at = datetime.combine(first, time.min, zone).astimezone(timezone.utc)
    end_at = datetime.combine(end, time.min, zone).astimezone(timezone.utc)
    visible = {p.project_id for p in runtime.project_store.list_visible_projects(actor, requester_is_admin=False)}
    items = []
    truncated = False
    # Bounded definition scan; report truncation instead of implying completeness.
    tasks = runtime.schedule_store.list_tasks(owner_user_id=actor, limit=1000)
    for task in tasks:
        if task.owner_user_id != actor or (task.project_id and task.project_id not in visible):
            continue
        executions = runtime.schedule_store.agenda_executions(scheduled_task_id=task.scheduled_task_id,
                                                              start_at=start_at.isoformat(), end_at=end_at.isoformat())
        truncated = truncated or len(executions) > 500
        observed = {}
        legacy = set()
        for execution in executions:
            prefix = task.scheduled_task_id + ":"
            raw = execution.occurrence_key.removeprefix(prefix) if execution.occurrence_key.startswith(prefix) else ""
            try:
                when = datetime.fromisoformat(raw).astimezone(timezone.utc)
            except ValueError:
                # Legacy rows have no reliable scheduled time. Show actual execution time explicitly.
                when = datetime.fromisoformat(execution.started_at).astimezone(timezone.utc)
                legacy.add(when)
            if start_at <= when < end_at:
                observed[when] = execution
        times = set(observed)
        cursor = start_at - timedelta(microseconds=1)
        definition = dict(task.schedule)
        if definition.get("kind") == "rrule" and not definition.get("dtstart"):
            definition["dtstart"] = task.created_at
        # dateutil seeks by iterating from DTSTART. Do not let old high-frequency
        # definitions turn a short read into millions of iteration steps.
        fields = dict(re.findall(r"(?:^|;)([A-Z]+)=([^;]+)", str(definition.get("rrule", ""))))
        seconds = {"SECONDLY": 1, "MINUTELY": 60, "HOURLY": 3600}.get(fields.get("FREQ"))
        expensive = False
        if seconds:
            anchor = datetime.fromisoformat(str(definition.get("dtstart", task.created_at))).astimezone(timezone.utc)
            interval = max(1, int(fields.get("INTERVAL", "1")))
            count = int(fields.get("COUNT", "100001"))
            expensive = min(count, max(0, (end_at - anchor).total_seconds()) / (seconds * interval)) > 100000
        if expensive:
            truncated = True
            if task.next_run_at:
                next_at = datetime.fromisoformat(task.next_run_at).astimezone(timezone.utc)
                if start_at <= next_at < end_at:
                    times.add(next_at)
        candidates = iter(())
        if not expensive:
            if definition.get("kind") == "rrule":
                anchor = _parse_datetime(str(definition["dtstart"]), timezone_name=task.timezone)
                rule = rrulestr(str(definition["rrule"]), dtstart=anchor.astimezone(ZoneInfo(task.timezone)))
                candidates = rule.xafter(cursor.astimezone(ZoneInfo(task.timezone)), inc=False)
            else:
                raw = compute_next_run_at(schedule=definition, timezone_name=task.timezone, after=cursor)
                candidates = iter([datetime.fromisoformat(raw)] if raw else [])
        for index, candidate in enumerate(islice(candidates, 501)):
            when = candidate.astimezone(timezone.utc)
            if when >= end_at:
                break
            if index == 500:
                truncated = True
                break
            if when >= start_at:
                times.add(when)
        for when in sorted(times):
            execution = observed.get(when)
            items.append({"id": f"{task.scheduled_task_id}:{when.isoformat()}", "scheduled_task_id": task.scheduled_task_id,
                          "project_id": task.project_id, "name": task.name, "at": when.isoformat(),
                          "day": when.astimezone(zone).date().isoformat(), "schedule_status": task.status,
                          "timezone": task.timezone, "recurrence": task.schedule,
                          "execution_status": execution.status if execution else None,
                          "run_id": execution.run_id if execution else None,
                          "kind": "execution" if when in legacy else "occurrence"})
            if len(items) >= 500:
                truncated = True
                break
        if len(items) >= 500:
            break
    result = {"items": sorted(items, key=lambda item: (item["at"], item["id"])), "timezone": str(zone),
              "start_day": start_day, "end_day": end_day, "truncated": truncated or len(tasks) == 1000,
              "unavailable_sources": []}
    if include_owner_calendar:
        status = _calendar_source_status(runtime, actor)
        if status == 'connected' and include_calendar_events:
            result.update(_calendar_events(runtime, actor, first, end, zone))
        elif status is not None:
            result["calendar_source_status"] = status
    return result
