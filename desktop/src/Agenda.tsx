import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, CalendarDays } from "lucide-react";
import { daemonRequest } from "./api";
import { agendaEntriesByDay, agendaRange, CALENDAR_CONFIGURATION_CHANGED, dayInZone, navigateAgenda, type AgendaEntry, type AgendaItem, type AgendaResult, type AgendaView, type CalendarEventStatus, type CalendarSourceStatus, type GoogleCalendarEvent } from "./agendaDates";

const calendarSourceCopy: Record<CalendarSourceStatus, { label: string; details: string }> = {
  connected: { label: "Google Calendar connected", details: "Shows events from your authorized Google calendar." },
  not_connected: { label: "Google Calendar not connected", details: "Connect Google Calendar in Settings → Integrations." },
  grant_required: { label: "Google Calendar needs calendar access", details: "Grant access to the calendar you want to use in Settings → Integrations → Google Calendar." },
  selection_required: { label: "Google Calendar needs a calendar selection", details: "Keep access enabled for one calendar in Settings → Integrations → Google Calendar." },
  reconnect_required: { label: "Google Calendar needs reconnection", details: "Reconnect your account in Settings → Integrations → Google Calendar." },
  unavailable: { label: "Google Calendar setup status unavailable", details: "Check the connection in Settings → Integrations → Google Calendar." },
};

const calendarEventCopy: Partial<Record<CalendarEventStatus, { label: string; details: string }>> = {
  partial: { label: "Google Calendar results limited", details: "Some events may be missing from this period." },
  unavailable: { label: "Google Calendar events unavailable", details: "Try reloading Agenda or check the connection in Settings." },
  api_disabled: { label: "Calendar MCP disabled", details: "Enable Calendar MCP in the Cloud project configured for Alphonse." },
  approval_required: { label: "Google Calendar approval required", details: "Google requires an additional approval to read these events." },
  credential_busy: { label: "Google Calendar connection busy", details: "Agenda retries automatically. If this persists, reconnect Google Calendar in Settings → Integrations." },
};

function AgendaSources({ result, compact }: { result: AgendaResult; compact?: boolean }) {
  const source = result.calendar_source_status === "connected" && result.calendar_event_status && calendarEventCopy[result.calendar_event_status]
    || (result.calendar_source_status ? calendarSourceCopy[result.calendar_source_status] : undefined);
  if (!source && !result.unavailable_sources.length) return null;
  const notes = result.unavailable_sources.map(text => <p className="agenda-source-note" key={text}>{text}</p>);
  if (compact) return <details className="widget-source-details"><summary>{source?.label || "Calendar sources"}</summary>{source && <p>{source.details}</p>}{notes}</details>;
  return <>{source && <p className="agenda-source-note"><strong>{source.label}.</strong> {source.details}</p>}{notes}</>;
}

type AgendaLoad = { scope: string; result: AgendaResult; error?: never } | { scope: string; error: string; result?: never };

export function Agenda({ member, timezone, active = true, compact = false, onSchedule, onProjects, onOpen, revision = 0 }: {
  member: string; timezone: string; active?: boolean; compact?: boolean; revision?: number;
  onSchedule: (id: string) => void; onProjects: () => void; onOpen?: () => void;
}) {
  const [view, setView] = useState<AgendaView>(compact ? "Today" : "Month");
  const dateScope = JSON.stringify([member, timezone]);
  const [selected, setSelected] = useState(() => ({ scope: dateScope, day: dayInZone(timezone) }));
  const day = selected.scope === dateScope ? selected.day : dayInZone(timezone);
  const setDay = (day: string) => setSelected({ scope: dateScope, day });
  const [loaded, setLoaded] = useState<AgendaLoad | null>(null);
  const [retry, setRetry] = useState(0);
  const range = agendaRange(day, view);
  const scope = JSON.stringify([member, timezone, range.start, range.end, retry]);
  const current = active && loaded?.scope === scope ? loaded : undefined;
  const result = current?.result;
  const error = current?.error;
  const loading = active && Boolean(member) && !current;
  useEffect(() => {
    if (!active || !member) {
      setLoaded(null);
      return;
    }
    setLoaded(previous => previous?.scope === scope ? previous : null);
    let live = true, generation = 0, loading = false, refetchOnFocus = false;
    const load = async (invalidate = false) => {
      if (loading && !invalidate) return;
      const request = ++generation;
      loading = true;
      refetchOnFocus = false;
      if (invalidate) setLoaded(null);
      try {
        const next = await daemonRequest<AgendaResult>("agenda", { actor_user_id: member, start_day: range.start, end_day: range.end, include_calendar_events: true });
        if (live && request === generation) setLoaded({ scope, result: next });
      } catch (cause) {
        if (live && request === generation) setLoaded({ scope, error: cause instanceof Error ? cause.message : "Agenda unavailable" });
      } finally {
        if (request === generation) {
          loading = false;
          if (live && refetchOnFocus) void load();
        }
      }
    };
    const refresh = () => void load();
    const changed = () => void load(true);
    const focused = () => { if (loading) refetchOnFocus = true; else refresh(); };
    refresh();
    const timer = window.setInterval(refresh, 30000);
    window.addEventListener(CALENDAR_CONFIGURATION_CHANGED, changed);
    window.addEventListener("focus", focused);
    return () => {
      live = false;
      window.clearInterval(timer);
      window.removeEventListener(CALENDAR_CONFIGURATION_CHANGED, changed);
      window.removeEventListener("focus", focused);
    };
  }, [member, scope, range.start, range.end, active, revision]);
  if (compact) return <section className="agenda agenda-compact" aria-label="Agenda preview">
    <div className="agenda-segments" role="group" aria-label="Agenda view">{(["Month", "Week", "Today"] as const).map(value => <button key={value} aria-pressed={view === value} onClick={() => { setView(value); if (value === "Today") setDay(dayInZone(timezone)); }}>{value}</button>)}</div>
    <div className="agenda-preview-navigation"><button aria-label="Previous period" title="Previous period" onClick={() => setDay(navigateAgenda(day, view, -1))}><ArrowLeft /></button><strong>{new Date(`${day}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", ...(view === "Month" ? { month: "long", year: "numeric" } : { weekday: "short", month: "short", day: "numeric" }) })}</strong><button aria-label="Next period" title="Next period" onClick={() => setDay(navigateAgenda(day, view, 1))}><ArrowRight /></button><details className="agenda-date-picker"><summary aria-label="Choose agenda date" title="Choose agenda date"><CalendarDays /></summary><label>Date <input aria-label="Agenda date" type="date" value={day} onChange={e => { if (e.target.value) setDay(e.target.value); }} /></label><button onClick={() => setDay(dayInZone(timezone))}>Return to today</button></details></div>
    {loading ? <p role="status">Loading agenda…</p> : error ? <div role="alert"><p>{error}</p><button onClick={() => setRetry(retry + 1)}>Retry agenda</button></div> : result && <><AgendaPreviewContent result={result} day={day} view={view} timezone={timezone} onDay={setDay} onSchedule={onSchedule} />{result.truncated && <p role="status">Results limited. Open Agenda to review a shorter period.</p>}<AgendaSources result={result} compact /></>}
    <footer><button onClick={onProjects}>Projects &amp; Tasks <ArrowRight aria-hidden="true" /></button>{onOpen && <button onClick={onOpen}>Detailed Agenda <ArrowRight aria-hidden="true" /></button>}</footer>
  </section>;
  const displayedDays = view === "Month"
    ? range.days.slice(0, Math.ceil((range.days.reduce((last, date, index) => date.slice(0, 7) === day.slice(0, 7) ? index : last, 0) + 1) / 7) * 7)
    : range.days;
  const entriesByDay = result && agendaEntriesByDay(result, displayedDays, timezone);
  const today = dayInZone(timezone);
  const period = new Date(`${day}T12:00:00Z`).toLocaleDateString(undefined, {
    timeZone: "UTC", ...(view === "Month" ? { month: "long", year: "numeric" } : view === "Week" ? { month: "long", year: "numeric" } : { weekday: "long", month: "long", day: "numeric" }),
  });
  return <section className="agenda agenda-workspace" aria-label="Alphonse’s Agenda">
    <header><h1>Alphonse’s Agenda</h1><p>The household’s time, with room for yours.</p></header>
    <div className="agenda-toolbar">
      <div className="agenda-period-controls"><h2>{period}</h2><div className="agenda-navigation" role="group" aria-label="Agenda date navigation"><button aria-label="Previous period" title="Previous period" onClick={() => setDay(navigateAgenda(day, view, -1))}><ArrowLeft /></button><button aria-label="Next period" title="Next period" onClick={() => setDay(navigateAgenda(day, view, 1))}><ArrowRight /></button><details className="agenda-workspace-picker"><summary aria-label="Choose agenda date" title="Choose agenda date"><CalendarDays /></summary><label>Date <input aria-label="Agenda date" type="date" value={day} onChange={e => { if (e.target.value) setDay(e.target.value); }} /></label><button onClick={() => setDay(dayInZone(timezone))}>Return to today</button></details></div></div>
      <div className="agenda-view-switch" role="group" aria-label="Agenda view">{(["Month", "Week", "Today"] as const).map(value => <button key={value} aria-pressed={view === value} onClick={() => { setView(value); if (value === "Today") setDay(dayInZone(timezone)); }}>{value}</button>)}</div>
    </div>
    {loading ? <p role="status">Loading agenda…</p> : error ? <div role="alert"><p>{error}</p><button onClick={() => setRetry(retry + 1)}>Retry agenda</button></div> : result && entriesByDay && <>
      {result.truncated && <p role="status">Results limited. Choose a shorter period to review more occurrences.</p>}
      {view === "Today" ? <AgendaTodayContent entries={entriesByDay.get(day) ?? []} day={day} timezone={timezone} onSchedule={onSchedule} calendarComplete={result.calendar_event_status === "complete"} /> : <div className="agenda-grid-scroll"><div className={`agenda-calendar agenda-calendar-${view.toLowerCase()}`}>
        <div className="agenda-weekdays">{["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map(label => <span key={label}>{label}</span>)}</div>
        <div className={`agenda-days agenda-${view.toLowerCase()}`}>{displayedDays.map(date => {
          const outside = view === "Month" && date.slice(0, 7) !== day.slice(0, 7);
          return <section key={date} className={`agenda-day${date === today ? " agenda-current" : ""}${outside ? " agenda-outside" : ""}`} aria-label={date}>
            {!outside && <><h3><time dateTime={date}>{view === "Month" ? Number(date.slice(-2)) : new Date(`${date}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", month: "short", day: "numeric" })}</time></h3>
            <AgendaRows entries={entriesByDay.get(date) ?? []} day={date} timezone={timezone} variant="grid" onSchedule={onSchedule} /></>}
          </section>;
        })}</div>
      </div></div>}
      <AgendaSources result={result} />
    </>}
    <footer><button onClick={onProjects}>Open Projects &amp; Tasks <ArrowRight aria-hidden="true" /></button>{onOpen && <button onClick={onOpen}>Detailed Agenda <ArrowRight aria-hidden="true" /></button>}</footer>
    <p className="agenda-help">Scheduled work belongs in the agenda and its related project or task. Past activity remains accessible from the item’s history.</p>
  </section>;
}

/** A chronological list of the selected day's events and scheduled work. */
export function AgendaTodayContent({ entries, day, timezone, onSchedule, calendarComplete }: {
  entries: AgendaEntry[]; day: string; timezone: string; onSchedule: (id: string) => void; calendarComplete?: boolean;
}) {
  return <div className="agenda-today-list" aria-label="Calendar events and scheduled work">
    {entries.length === 0 ? <p className="agenda-today-empty" role="status">{calendarComplete ? "No events or scheduled work on this day." : "No scheduled work on this day."}</p>
      : <AgendaRows entries={entries} day={day} timezone={timezone} variant="today" onSchedule={onSchedule} />}
  </div>;
}

/** A glanceable preview, sharing the workspace's bounded authorized source. */
export function AgendaPreviewContent({ result, day, view, timezone, onDay, onSchedule }: {
  result: AgendaResult; day: string; view: AgendaView; timezone: string;
  onDay: (day: string) => void; onSchedule: (id: string) => void;
}) {
  const range = agendaRange(day, view);
  const days = view === "Week" ? range.days : [day];
  const entriesByDay = agendaEntriesByDay(result, range.days, timezone);
  return <>
    {view === "Month" && <div className="agenda-mini-month" aria-label="Month dates">
      {["M", "T", "W", "T", "F", "S", "S"].map((label, i) => <span key={i} aria-hidden="true">{label}</span>)}
      {range.days.map(date => {
        const entries = entriesByDay.get(date) ?? [];
        const sources = [entries.some(entry => entry.source === "google") && "Google Calendar events",
          entries.some(entry => entry.source === "schedule") && "scheduled work"].filter(Boolean);
        return <button key={date} className={`${date.slice(0, 7) !== day.slice(0, 7) ? "outside " : ""}${entries.length ? "has-occurrences" : ""}`} aria-label={sources.length ? `${date}: ${sources.join(" and ")}` : date} aria-pressed={date === day} onClick={() => onDay(date)}>{Number(date.slice(-2))}</button>;
      })}
    </div>}
    <div className="agenda-preview-events">{days.map(date => {
      const entries = entriesByDay.get(date) ?? [];
      return <section key={date} aria-label={date}>{view !== "Today" && <h4>{new Date(`${date}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" })}</h4>}
        {!entries.length && <p className="agenda-preview-empty">{result.calendar_event_status === "complete" ? "No events or scheduled work." : "No scheduled work."}</p>}
        <AgendaRows entries={entries} day={date} timezone={timezone} variant="preview" onSchedule={onSchedule} />
      </section>;
    })}</div>
  </>;
}

type RowVariant = "grid" | "today" | "preview";

function AgendaRows({ entries, day, timezone, variant, onSchedule }: {
  entries: AgendaEntry[]; day: string; timezone: string; variant: RowVariant; onSchedule: (id: string) => void;
}) {
  return entries.map(({ source, item }) => source === "google"
    ? <GoogleEvent key={`google:${item.id}`} event={item} day={day} timezone={timezone} variant={variant} />
    : <ScheduledEvent key={`schedule:${item.id}`} item={item} timezone={timezone} variant={variant} onSchedule={onSchedule} />);
}

function timeLabel(value: string, timezone: string, variant: RowVariant) {
  return new Date(value).toLocaleTimeString(undefined, { timeZone: timezone, hour: variant === "today" ? "numeric" : "2-digit",
    minute: "2-digit", ...(variant === "preview" ? {} : { hour12: false }) });
}

function ScheduledEvent({ item, timezone, variant, onSchedule }: {
  item: AgendaItem; timezone: string; variant: RowVariant; onSchedule: (id: string) => void;
}) {
  const clock = <time dateTime={item.at}>{timeLabel(item.at, timezone, variant)}</time>;
  const open = () => onSchedule(item.scheduled_task_id);
  if (variant === "grid") return <button className="agenda-occurrence" onClick={open}
    title={`${item.name} · ${item.schedule_status}${item.execution_status ? ` · ${item.execution_status}` : ""}`}>{clock}{" "}<span>{item.name}</span></button>;
  const kind = item.kind === "execution" ? "Actual execution" : "Scheduled";
  if (variant === "today") return <button className="agenda-today-event" onClick={open}>
    {clock}<span className="agenda-today-details"><strong>{item.name}</strong><small>{kind} · {item.schedule_status.replaceAll("_", " ")}{item.execution_status ? ` · ${item.execution_status.replaceAll("_", " ")}` : ""}</small></span>
  </button>;
  return <button className="agenda-preview-event" onClick={open}>{clock}<span><strong>{item.name}</strong><small>{kind} · {item.schedule_status}{item.execution_status ? ` · ${item.execution_status}` : " · not run"}</small>{item.timezone !== timezone && <small>{item.timezone}</small>}</span></button>;
}

function GoogleEvent({ event, day, timezone, variant }: { event: GoogleCalendarEvent; day: string; timezone: string; variant: RowVariant }) {
  const startsToday = !event.all_day && dayInZone(timezone, new Date(event.start)) === day;
  const endsToday = !event.all_day && !event.end_unspecified && dayInZone(timezone, new Date(event.end)) === day;
  const clock = event.all_day ? <time dateTime={day}>All day</time>
    : startsToday ? <time dateTime={event.start}>{timeLabel(event.start, timezone, variant)}</time>
    : endsToday ? <time dateTime={event.end}>Until {timeLabel(event.end, timezone, variant)}</time>
    : <span className="agenda-continuing-event">Ongoing</span>;
  const title = event.title || "Untitled event";
  if (variant === "grid") return <div className="agenda-occurrence agenda-google-event">{clock}{" "}<span>{title}</span><small>Google Calendar</small></div>;
  return <div className={`agenda-${variant}-event agenda-google-event`}>{clock}
    <span className={variant === "today" ? "agenda-today-details" : undefined}><strong>{title}</strong>
      <small>Google Calendar{startsToday && endsToday ? ` · Until ${timeLabel(event.end, timezone, variant)}` : ""}</small>
    </span>
  </div>;
}
