import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, CalendarDays } from "lucide-react";
import { daemonRequest } from "./api";
import { agendaRange, dayInZone, navigateAgenda, type AgendaItem, type AgendaResult, type AgendaView } from "./agendaDates";

export function Agenda({ member, timezone, active = true, compact = false, onSchedule, onProjects, onOpen, revision = 0 }: {
  member: string; timezone: string; active?: boolean; compact?: boolean; revision?: number;
  onSchedule: (id: string) => void; onProjects: () => void; onOpen?: () => void;
}) {
  const [view, setView] = useState<AgendaView>(compact ? "Today" : "Month");
  const [day, setDay] = useState(() => dayInZone(timezone));
  const [result, setResult] = useState<AgendaResult | null>(null);
  const [error, setError] = useState(""); const [loading, setLoading] = useState(true); const [retry, setRetry] = useState(0);
  const range = agendaRange(day, view);
  useEffect(() => { setDay(dayInZone(timezone)); }, [timezone, member]);
  useEffect(() => {
    let live = true; let generation = 0;
    if (!active || !member) return;
    const load = async () => {
      const request = ++generation;
      setLoading(true); setResult(null); setError("");
      try { const next = await daemonRequest<AgendaResult>("agenda", { actor_user_id: member, start_day: range.start, end_day: range.end }); if (live && request === generation) setResult(next); }
      catch (cause) { if (live && request === generation) setError(cause instanceof Error ? cause.message : "Agenda unavailable"); }
      finally { if (live && request === generation) setLoading(false); }
    };
    void load(); const timer = window.setInterval(() => void load(), 30000);
    return () => { live = false; window.clearInterval(timer); };
  }, [member, timezone, range.start, range.end, retry, active, revision]);
  if (compact) return <section className="agenda agenda-compact" aria-label="Agenda preview">
    <div className="agenda-segments" role="group" aria-label="Agenda view">{(["Month", "Week", "Today"] as const).map(value => <button key={value} aria-pressed={view === value} onClick={() => { setView(value); if (value === "Today") setDay(dayInZone(timezone)); }}>{value}</button>)}</div>
    <div className="agenda-preview-navigation"><button aria-label="Previous period" title="Previous period" onClick={() => setDay(navigateAgenda(day, view, -1))}><ArrowLeft /></button><strong>{new Date(`${day}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", ...(view === "Month" ? { month: "long", year: "numeric" } : { weekday: "short", month: "short", day: "numeric" }) })}</strong><button aria-label="Next period" title="Next period" onClick={() => setDay(navigateAgenda(day, view, 1))}><ArrowRight /></button><details className="agenda-date-picker"><summary aria-label="Choose agenda date" title="Choose agenda date"><CalendarDays /></summary><label>Date <input aria-label="Agenda date" type="date" value={day} onChange={e => { if (e.target.value) setDay(e.target.value); }} /></label><button onClick={() => setDay(dayInZone(timezone))}>Return to today</button></details></div>
    {loading ? <p role="status">Loading agenda…</p> : error ? <div role="alert"><p>{error}</p><button onClick={() => setRetry(retry + 1)}>Retry agenda</button></div> : result && <><AgendaPreviewContent result={result} day={day} view={view} timezone={timezone} onDay={setDay} onSchedule={onSchedule} />{result.truncated && <p role="status">Results limited. Open Agenda to review a shorter period.</p>}{result.unavailable_sources.length > 0 && <details className="widget-source-details"><summary>Calendar sources</summary>{result.unavailable_sources.map(source => <p key={source}>{source}</p>)}</details>}</>}
    <footer><button onClick={onProjects}>Projects &amp; Tasks <ArrowRight aria-hidden="true" /></button>{onOpen && <button onClick={onOpen}>Detailed Agenda <ArrowRight aria-hidden="true" /></button>}</footer>
  </section>;
  const displayedDays = view === "Month"
    ? range.days.slice(0, Math.ceil((range.days.reduce((last, date, index) => date.slice(0, 7) === day.slice(0, 7) ? index : last, 0) + 1) / 7) * 7)
    : range.days;
  const period = new Date(`${day}T12:00:00Z`).toLocaleDateString(undefined, {
    timeZone: "UTC", ...(view === "Month" ? { month: "long", year: "numeric" } : view === "Week" ? { month: "long", year: "numeric" } : { weekday: "long", month: "long", day: "numeric" }),
  });
  return <section className="agenda agenda-workspace" aria-label="Alphonse’s Agenda">
    <header><h1>Alphonse’s Agenda</h1><p>The household’s time, with room for yours.</p></header>
    <div className="agenda-toolbar">
      <div className="agenda-period-controls"><h2>{period}</h2><div className="agenda-navigation" role="group" aria-label="Agenda date navigation"><button aria-label="Previous period" title="Previous period" onClick={() => setDay(navigateAgenda(day, view, -1))}><ArrowLeft /></button><button aria-label="Next period" title="Next period" onClick={() => setDay(navigateAgenda(day, view, 1))}><ArrowRight /></button><details className="agenda-workspace-picker"><summary aria-label="Choose agenda date" title="Choose agenda date"><CalendarDays /></summary><label>Date <input aria-label="Agenda date" type="date" value={day} onChange={e => { if (e.target.value) setDay(e.target.value); }} /></label><button onClick={() => setDay(dayInZone(timezone))}>Return to today</button></details></div></div>
      <div className="agenda-view-switch" role="group" aria-label="Agenda view">{(["Month", "Week", "Today"] as const).map(value => <button key={value} aria-pressed={view === value} onClick={() => { setView(value); if (value === "Today") setDay(dayInZone(timezone)); }}>{value}</button>)}</div>
    </div>
    {loading ? <p role="status">Loading agenda…</p> : error ? <div role="alert"><p>{error}</p><button onClick={() => setRetry(retry + 1)}>Retry agenda</button></div> : result && <>
      {result.truncated && <p role="status">Results limited. Choose a shorter period to review more occurrences.</p>}
      {view === "Today" ? <AgendaTodayContent items={result.items.filter(item => item.day === day)} timezone={timezone} onSchedule={onSchedule} /> : <div className="agenda-grid-scroll"><div className={`agenda-calendar agenda-calendar-${view.toLowerCase()}`}>
        <div className="agenda-weekdays">{["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map(label => <span key={label}>{label}</span>)}</div>
        <div className={`agenda-days agenda-${view.toLowerCase()}`}>{displayedDays.map(date => {
          const outside = view === "Month" && date.slice(0, 7) !== day.slice(0, 7);
          return <section key={date} className={`agenda-day${date === dayInZone(timezone) ? " agenda-current" : ""}${outside ? " agenda-outside" : ""}`} aria-label={date}>
            {!outside && <><h3><time dateTime={date}>{view === "Month" ? Number(date.slice(-2)) : new Date(`${date}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", month: "short", day: "numeric" })}</time></h3>
            {result.items.filter(item => item.day === date).map(item => <button className="agenda-occurrence" key={item.id} onClick={() => onSchedule(item.scheduled_task_id)} title={`${item.name} · ${item.schedule_status}${item.execution_status ? ` · ${item.execution_status}` : ""}`}><time dateTime={item.at}>{new Date(item.at).toLocaleTimeString(undefined, { timeZone: timezone, hour: "2-digit", minute: "2-digit", hour12: false })}</time>{" "}<span>{item.name}</span></button>)}</>}
          </section>;
        })}</div>
      </div></div>}
      {result.unavailable_sources.map(source => <p className="agenda-source-note" key={source}>{source}</p>)}
    </>}
    <footer><button onClick={onProjects}>Open Projects &amp; Tasks <ArrowRight aria-hidden="true" /></button>{onOpen && <button onClick={onOpen}>Detailed Agenda <ArrowRight aria-hidden="true" /></button>}</footer>
    <p className="agenda-help">Scheduled work belongs in the agenda and its related project or task. Past activity remains accessible from the item’s history.</p>
  </section>;
}

/** A chronological list of the selected day's scheduled work. */
export function AgendaTodayContent({ items, timezone, onSchedule }: {
  items: AgendaItem[]; timezone: string; onSchedule: (id: string) => void;
}) {
  return <div className="agenda-today-list" aria-label="Scheduled work">
    {items.length === 0 ? <p className="agenda-today-empty" role="status">No scheduled work on this day.</p> : items.map(item =>
      <button className="agenda-today-event" key={item.id} onClick={() => onSchedule(item.scheduled_task_id)}>
        <time dateTime={item.at}>{new Date(item.at).toLocaleTimeString(undefined, { timeZone: timezone, hour: "numeric", minute: "2-digit", hour12: false })}</time>
        <span className="agenda-today-details"><strong>{item.name}</strong><small>{item.kind === "execution" ? "Actual execution" : "Scheduled"} · {item.schedule_status.replaceAll("_", " ")}{item.execution_status ? ` · ${item.execution_status.replaceAll("_", " ")}` : ""}</small></span>
      </button>
    )}
  </div>;
}

/** A glanceable preview, sharing the workspace's bounded authorized source. */
export function AgendaPreviewContent({ result, day, view, timezone, onDay, onSchedule }: {
  result: AgendaResult; day: string; view: AgendaView; timezone: string;
  onDay: (day: string) => void; onSchedule: (id: string) => void;
}) {
  const range = agendaRange(day, view);
  const days = view === "Week" ? range.days : [day];
  return <>
    {view === "Month" && <div className="agenda-mini-month" aria-label="Month dates">
      {["M", "T", "W", "T", "F", "S", "S"].map((label, i) => <span key={i} aria-hidden="true">{label}</span>)}
      {range.days.map(date => <button key={date} className={`${date.slice(0, 7) !== day.slice(0, 7) ? "outside " : ""}${result.items.some(item => item.day === date) ? "has-occurrences" : ""}`} aria-label={`${date}${result.items.some(item => item.day === date) ? ": scheduled work" : ""}`} aria-pressed={date === day} onClick={() => onDay(date)}>{Number(date.slice(-2))}</button>)}
    </div>}
    <div className="agenda-preview-events">{days.map(date => {
      const items = result.items.filter(item => item.day === date);
      return <section key={date} aria-label={date}>{view !== "Today" && <h4>{new Date(`${date}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" })}</h4>}
        {!items.length && <p className="agenda-preview-empty">No scheduled work.</p>}
        {items.map(item => <button className="agenda-preview-event" key={item.id} onClick={() => onSchedule(item.scheduled_task_id)}><time dateTime={item.at}>{new Date(item.at).toLocaleTimeString(undefined, { timeZone: timezone, hour: "2-digit", minute: "2-digit" })}</time><span><strong>{item.name}</strong><small>{item.kind === "execution" ? "Actual execution" : "Scheduled"} · {item.schedule_status}{item.execution_status ? ` · ${item.execution_status}` : " · not run"}</small>{item.timezone !== timezone && <small>{item.timezone}</small>}</span></button>)}
      </section>;
    })}</div>
  </>;
}
