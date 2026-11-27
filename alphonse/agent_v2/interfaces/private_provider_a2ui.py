"""Inert A2UI cards for verified private Desktop pickup."""
from datetime import date, datetime, time, timedelta
from email.utils import parseaddr
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


def _header(components, source, title, *, partial, coverage_note):
    heading = ["title"]
    components.extend([_text("source", source),
                       _text("title", title, variant="heading"),
                       {"id": "result_head", "component": "Row", "children": heading}])
    if partial:
        heading.extend(["coverage", "coverage_note"])
        components.extend([_text("coverage", "Partial coverage", component="Status"),
                           _text("coverage_note", coverage_note)])


def gmail_summary_surface(message_id, observation, *, source, summary, timezone_name=None, searches=()):
    """Project the private digest and literal, bounded excerpts for its citations."""
    messages = observation["messages"]
    days = observation["lookback_days"]
    count = len(messages)
    scope = f'{count} {"message" if count == 1 else "messages"} reviewed · Past {days} {"day" if days == 1 else "days"}'
    digest = parse_gmail_digest(summary, count)
    title = digest.title or ("Deadlines & next steps" if digest.actions else "Gmail summary")
    components = [_text("scope", scope)]
    _header(components, source.replace("Gmail label ", "Gmail · ", 1), title,
            partial=observation["status"] == "partial", coverage_note="A narrower search may find more messages.")
    children = ["source", "result_head", "scope"]
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
            title_id, meta_id = f"{row}_title", f"{row}_meta"
            citations = [f"{row}_cite_{number}" for number in task.sources]
            metadata = [f"{row}_detail"] if task.detail else []
            components.extend([
                {"id": row, "component": "Row", "children": [f"{row}_when", f"{row}_body"]},
                _text(f"{row}_when", task.when or "No date given", variant="due" if task.when else None),
                {"id": f"{row}_body", "component": "Column", "children": [title_id, meta_id]},
                _text(title_id, task.title, variant="strong"),
                {"id": meta_id, "component": "Row", "children": [*metadata, *citations]},
            ])
            if metadata:
                components.append(_text(metadata[0], task.detail, component="Markdown"))
            components.extend({"id": citation, "component": "EmailCitation", "sourceNumber": number}
                              for citation, number in zip(citations, task.sources))
    if digest.summary:
        components.append(_text("findings", digest.summary, component="Markdown"))
        children.append("findings")
    if not digest.actions and not digest.summary:
        components.append(_text("empty", "A structured summary could not be displayed. The reviewed emails are below."))
        children.append("empty")
    cited = digest.source_order(count)
    superseded = digest.superseded_sources()
    zone = _zone(timezone_name, fallback="UTC")
    indexes = cited or range(1, count + 1)
    source_ids = []
    for index in indexes:
        message = messages[index - 1]
        row = f"source_{index}"
        source_ids.append(row)
        subject = message["subject"] or "No subject"
        sender_name, sender_address = parseaddr(message["sender"])
        sender = sender_name or sender_address or message["sender"] or "Sender unavailable"
        received = datetime.fromisoformat(message["received_at"]).astimezone(zone).date()
        details = [f"{row}_title", f"{row}_meta"]
        annotations = [sender, _month_day(received)]
        if index in superseded:
            annotations.append("Superseded")
        if index in cited and message.get("excerpt_truncated"):
            annotations.append("Excerpt shortened")
        components.extend([
            {"id": row, "component": "Column", "children": details},
            _text(details[0], f"[{index}] {subject}", variant="strong"),
            _text(details[1], " · ".join(annotations)),
        ])
        if index in cited and message.get("excerpt"):
            details.append(f"{row}_excerpt")
            components.append(_text(details[-1], message["excerpt"], variant="quote"))
    source_label = f"Cited emails ({len(cited)} of {count} reviewed)" if cited else f"Reviewed emails ({count})"
    components.extend([
        {"id": "sources", "component": "Disclosure", "label": source_label, "children": ["source_list", "qualifier"]},
        {"id": "source_list", "component": "List", "children": source_ids},
        _text("qualifier", "Based on message excerpts. Attachments were omitted."),
    ])
    return _surface(message_id, "gmail", components, [*children, "sources"])


def _month_day(day):
    return f"{day:%b} {day.day}"


def _date_label(day):
    return f"{day:%A}, {_month_day(day)}"


def _zone(name, *, fallback):
    try:
        return ZoneInfo(name)
    except (TypeError, ZoneInfoNotFoundError):
        return ZoneInfo(fallback)


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
        return f"{_month_day(first)} schedule"
    ending = str(last.day) if first.month == last.month and first.year == last.year else _month_day(last)
    return f"{_month_day(first)}–{ending} schedule"


def calendar_agenda_surface(message_id, observation, *, timezone_name):
    """Group timed events by the Desktop user's local date. Keep all-day dates as given."""
    zone = _zone(timezone_name, fallback=observation["timezone"])
    start, end = (datetime.fromisoformat(value).astimezone(zone) for value in observation["window"])
    city = zone.key.rsplit("/", 1)[-1].replace("_", " ")
    scope = f"{city} time"
    if start.time() != time.min or end.time() != time.min:
        scope += f" · {_date_label(start.date())}, {start:%H:%M} – {_date_label(end.date())}, {end:%H:%M}"
    components = [_text("scope", scope)]
    _header(components, "Google Calendar", _agenda_title(start, end),
            partial=observation["status"] == "partial", coverage_note="Try a shorter date range.")
    children = ["source", "result_head", "scope"]
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
            _text(f"{group}_heading", f"{day:%a}, {_month_day(day)}", variant="subheading"),
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
