"""Inert A2UI cards for verified private Desktop pickup."""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alphonse.agent_v2.core.intelligence.simple_loop.gmail_digest import parse_gmail_digest
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail_search import gmail_earlier_empty_scopes

from .a2ui import ALPHONSE_DESKTOP_CATALOG_ID, A2UI_VERSION, validate_envelope


def _text(identifier, text, *, component="Text", variant=None):
    item = {"id": identifier, "component": component, "text": text}
    if variant:
        item["variant"] = variant
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


def gmail_summary_surface(message_id, observation, *, source, summary, searches=()):
    """Project the private digest and literal, bounded excerpts for its citations."""
    messages = observation["messages"]
    days = observation["lookback_days"]
    count = len(messages)
    scope = f'{count} {"message" if count == 1 else "messages"} reviewed · Past {days} {"day" if days == 1 else "days"}'
    digest = parse_gmail_digest(summary, count)
    title = "Deadlines & next steps" if digest.actions else "Gmail summary"
    components = [_text("source", source.replace("Gmail label ", "Gmail · ", 1)),
                  _text("title", title, variant="heading"), _text("scope", scope)]
    children = ["source", "title", "scope"]
    if observation["status"] == "partial":
        components.extend([_text("coverage", "Partial coverage", component="Status"),
                           _text("coverage_note", "A narrower search may find more messages.")])
        children.extend(["coverage", "coverage_note"])
    if not messages:
        components.append(_text("empty", "Gmail returned no matching messages in this search."
                                if observation["status"] == "complete" else "No messages could be reviewed."))
        children.append("empty")
        for index, scope in enumerate(gmail_earlier_empty_scopes(observation, searches), 1):
            key = f"earlier_scope_{index}"
            components.append(_text(key, f"Also searched {scope}."))
            children.append(key)
        return _surface(message_id, "gmail", components, children)
    if digest.actions:
        rows = []
        components.append({"id": "actions", "component": "List", "children": rows})
        children.append("actions")
        for index, task in enumerate(digest.actions, 1):
            row = f"action_{index}"
            rows.append(row)
            details = [f"{row}_title"]
            citations = [f"{row}_cite_{number}" for number in task.sources]
            components.extend([
                {"id": row, "component": "Row", "children": [f"{row}_when", f"{row}_body"]},
                _text(f"{row}_when", task.when or "No date given", variant="due" if task.when else None),
                {"id": f"{row}_body", "component": "Column", "children": details},
                _text(details[0], task.title, variant="strong"),
            ])
            if task.detail:
                details.append(f"{row}_detail")
                components.append(_text(details[-1], task.detail, component="Markdown"))
            details.append(f"{row}_citations")
            components.append({"id": details[-1], "component": "Row", "children": citations})
            components.extend({"id": citation, "component": "EmailCitation", "sourceNumber": number}
                              for citation, number in zip(citations, task.sources))
    if digest.summary:
        components.append(_text("findings", digest.summary, component="Markdown"))
        children.append("findings")
    if not digest.actions and not digest.summary:
        components.append(_text("empty", "A structured summary could not be displayed. The reviewed emails are below."))
        children.append("empty")
    cited = digest.cited_sources(count)
    indexes = sorted(cited) if cited else range(1, count + 1)
    source_ids = []
    for index in indexes:
        message = messages[index - 1]
        row = f"source_{index}"
        source_ids.append(row)
        subject = message["subject"] or "No subject"
        sender = message["sender"] or "Sender unavailable"
        received = datetime.fromisoformat(message["received_at"]).date().isoformat()
        details = [f"{row}_title", f"{row}_meta"]
        shortened = " · Excerpt shortened" if index in cited and message.get("excerpt_truncated") else ""
        components.extend([
            {"id": row, "component": "Column", "children": details},
            _text(details[0], f"[{index}] {subject}", variant="strong"),
            _text(details[1], f"{sender} · {received} UTC{shortened}"),
        ])
        if index in cited and message.get("excerpt"):
            details.append(f"{row}_excerpt")
            components.append(_text(details[-1], message["excerpt"], variant="quote"))
    source_label = f"Cited emails ({len(cited)} of {count} reviewed)" if cited else f"Reviewed emails ({count})"
    components.extend([
        {"id": "sources", "component": "Disclosure", "label": source_label, "children": ["source_list"]},
        {"id": "source_list", "component": "List", "children": source_ids},
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
        detail = f"Continues through {_date_label(last)}" if start != last else ""
        return start, (0, 0), "All day", detail
    start, end = (datetime.fromisoformat(event[key]).astimezone(zone) for key in ("start", "end"))
    when = start.strftime("%H:%M")
    detail = ""
    if event["end_unspecified"]:
        detail = "End time unspecified"
    elif start.date() == end.date():
        when += f"–{end:%H:%M}"
    else:
        detail = f"Ends {_date_label(end.date())}, {end:%H:%M}"
    return start.date(), (1, start.timestamp()), when, detail


def _agenda_title(start, end):
    first, last = start.date(), end.date()
    if end.time() == time.min:
        last -= timedelta(days=1)
    if first == last:
        return f"{first:%b} {first.day} agenda"
    ending = str(last.day) if first.month == last.month and first.year == last.year else f"{last:%b} {last.day}"
    return f"{first:%b} {first.day}–{ending} agenda"


def calendar_agenda_surface(message_id, observation, *, timezone_name):
    """Group timed events by the Desktop user's local date. Keep all-day dates as given."""
    try:
        zone = ZoneInfo(timezone_name)
    except (TypeError, ZoneInfoNotFoundError):
        zone = ZoneInfo(observation["timezone"])
    start, end = (datetime.fromisoformat(value).astimezone(zone) for value in observation["window"])
    components = [_text("source", "Google Calendar"),
                  _text("title", _agenda_title(start, end), variant="heading"),
                  _text("scope", f"{_range_label(start, end)} · {zone.key}")]
    children = ["source", "title", "scope"]
    if observation["status"] == "partial":
        components.extend([_text("coverage", "Partial coverage", component="Status"),
                           _text("coverage_note", "Try a shorter date range.")])
        children.extend(["coverage", "coverage_note"])
    groups = {}
    for event in observation["events"]:
        day, order, when, detail = _calendar_event_time(event, zone)
        if day < start.date() and not event["all_day"]:
            detail = f"Started {_date_label(day)} at {when}. {detail}".strip()
            when = "Ongoing"
        groups.setdefault(max(day, start.date()), []).append((order, when, event["title"] or "Title unavailable", detail))
    for index, (day, events) in enumerate(sorted(groups.items()), 1):
        group = f"day_{index}"
        children.append(group)
        rows = []
        components.extend([
            {"id": group, "component": "Column", "children": [f"{group}_heading", f"{group}_events"]},
            _text(f"{group}_heading", _date_label(day), variant="subheading"),
            {"id": f"{group}_events", "component": "List", "children": rows},
        ])
        for number, (_, when, title, detail) in enumerate(sorted(events), 1):
            row = f"{group}_event_{number}"
            rows.append(row)
            content = [f"{row}_title"]
            components.extend([
                {"id": row, "component": "Row", "children": [f"{row}_time", f"{row}_body"]},
                _text(f"{row}_time", when),
                {"id": f"{row}_body", "component": "Column", "children": content},
                _text(content[0], title, variant="strong"),
            ])
            if detail:
                content.append(f"{row}_meta")
                components.append(_text(content[-1], detail))
    if not groups:
        components.append(_text("empty", "No events were returned in this window."
                                if observation["status"] == "complete" else "No events could be verified in this page."))
        children.append("empty")
    return _surface(message_id, "calendar", components, children)
