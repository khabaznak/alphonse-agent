"""Inert A2UI cards for verified private Desktop pickup."""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .a2ui import ALPHONSE_DESKTOP_CATALOG_ID, A2UI_VERSION, validate_envelope


def _text(id, text, *, component="Text", heading=False):
    item = {"id": id, "component": component, "text": text}
    if heading:
        item["variant"] = "heading"
    return item


def _surface(message_id, provider, components, children):
    surface_id = f"private-{provider}:{message_id}"
    envelopes = [
        {"version": A2UI_VERSION, "createSurface": {
            "surfaceId": surface_id, "catalogId": ALPHONSE_DESKTOP_CATALOG_ID, "sendDataModel": False}},
        {"version": A2UI_VERSION, "updateComponents": {
            "surfaceId": surface_id, "components": [
                {"id": "root", "component": "Card", "children": children}, *components]}},
    ]
    for envelope in envelopes:
        validate_envelope(envelope)
    return envelopes


def gmail_summary_surface(message_id, observation, *, source, summary):
    """Show the synthesis and the numbered source metadata without message bodies."""
    messages = observation["messages"]
    days = observation["lookback_days"]
    count = len(messages)
    scope = f'{count} {"message" if count == 1 else "messages"} reviewed · Past {days} {"day" if days == 1 else "days"}'
    components = [_text("source", source, heading=True), _text("scope", scope)]
    children = ["source", "scope"]
    if observation["status"] == "partial":
        components.append(_text("coverage", "Partial coverage · A narrower search may find more messages.",
                                component="Status"))
        children.append("coverage")
    if not messages:
        components.append(_text("empty", "Gmail returned no matching messages in this search."
                                if observation["status"] == "complete" else "No messages could be reviewed."))
        return _surface(message_id, "gmail", components, [*children, "empty"])
    if summary:
        components.append(_text("findings", summary))
        children.append("findings")
    source_ids = []
    for index, message in enumerate(messages, 1):
        row = f"source_{index}"
        source_ids.append(row)
        subject = message["subject"] or "No subject"
        sender = message["sender"] or "Sender unavailable"
        received = datetime.fromisoformat(message["received_at"]).date().isoformat()
        components.extend([
            {"id": row, "component": "Column", "children": [f"{row}_title", f"{row}_meta"]},
            _text(f"{row}_title", f"[{index}] {subject}"),
            _text(f"{row}_meta", f"{sender} · {received} UTC"),
        ])
    components.extend([
        {"id": "sources", "component": "Disclosure", "label": "Reviewed sources", "children": source_ids},
        _text("qualifier", "Based on message excerpts. Attachments were omitted."),
    ])
    return _surface(message_id, "gmail", components, [*children, "sources", "qualifier"])


def _date_label(day):
    return day.strftime("%a, %d %b %Y")


def _range_label(start, end):
    if start.time() == time.min and end.time() == time.min:
        last_day = end.date() - timedelta(days=1)
        return _date_label(start.date()) if start.date() == last_day else f"{_date_label(start.date())} – {_date_label(last_day)}"
    return f'{start:%d %b %Y, %H:%M} – {end:%d %b %Y, %H:%M}'


def _calendar_event_time(event, zone):
    if event["all_day"]:
        start = date.fromisoformat(event["start"])
        last = date.fromisoformat(event["end"]) - timedelta(days=1)
        return start, "All day" if start == last else f"All day · Through {_date_label(last)}"
    start, end = (datetime.fromisoformat(event[key]).astimezone(zone) for key in ("start", "end"))
    when = start.strftime("%H:%M")
    if not event["end_unspecified"]:
        when += f"–{end:%H:%M}" if start.date() == end.date() else f" – {_date_label(end.date())}, {end:%H:%M}"
    return start.date(), when


def calendar_agenda_surface(message_id, observation, *, timezone_name):
    """Group timed events by the Desktop user's local date. Keep all-day dates as given."""
    try:
        zone = ZoneInfo(timezone_name)
    except (TypeError, ZoneInfoNotFoundError):
        zone = ZoneInfo(observation["timezone"])
    start, end = (datetime.fromisoformat(value).astimezone(zone) for value in observation["window"])
    components = [_text("source", "Google Calendar", heading=True),
                  _text("scope", f"{_range_label(start, end)} · {zone.key}")]
    children = ["source", "scope"]
    if observation["status"] == "partial":
        components.append(_text("coverage", "Partial coverage · Try a shorter date range.", component="Status"))
        children.append("coverage")
    groups = {}
    for event in observation["events"]:
        day, when = _calendar_event_time(event, zone)
        groups.setdefault(day, []).append((when, event["title"] or "Title unavailable"))
    for index, (day, events) in enumerate(sorted(groups.items()), 1):
        group = f"day_{index}"
        children.append(group)
        rows = [f"{group}_heading"]
        components.extend([
            {"id": group, "component": "Column", "children": rows},
            _text(rows[0], _date_label(day), heading=True),
        ])
        for number, (when, title) in enumerate(events, 1):
            row = f"{group}_event_{number}"
            rows.append(row)
            components.extend([
                {"id": row, "component": "Row", "children": [f"{row}_time", f"{row}_title"]},
                _text(f"{row}_time", when),
                _text(f"{row}_title", title),
            ])
    if not groups:
        components.append(_text("empty", "No events were returned in this window."
                                if observation["status"] == "complete" else "No events could be verified in this page."))
        children.append("empty")
    return _surface(message_id, "calendar", components, children)
