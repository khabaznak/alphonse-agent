import { useEffect, useState } from "react";
import { daemonRequest } from "./api";
import { agendaRange, dayInZone, navigateAgenda, type AgendaResult, type AgendaView } from "./agendaDates";

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
  return <section className={`agenda${compact ? " agenda-compact" : ""}`} aria-label={compact ? "Agenda preview" : "Alphonse’s Agenda"}>
    {!compact && <header><h1>Alphonse’s Agenda</h1><p>When scheduled work happens · {timezone}</p></header>}
    <div className="agenda-toolbar"><div role="group" aria-label="Agenda view">{(["Month", "Week", "Today"] as const).map(value => <button key={value} aria-pressed={view === value} onClick={() => { setView(value); if (value === "Today") setDay(dayInZone(timezone)); }}>{value}</button>)}</div><div role="group" aria-label="Agenda date navigation"><button aria-label="Previous period" onClick={() => setDay(navigateAgenda(day, view, -1))}>←</button><button onClick={() => setDay(dayInZone(timezone))}>Today</button><button aria-label="Next period" onClick={() => setDay(navigateAgenda(day, view, 1))}>→</button></div><label>Date <input aria-label="Agenda date" type="date" value={day} onChange={e => { if (e.target.value) setDay(e.target.value); }} /></label></div>
    <p className="agenda-period">{day.slice(0, 7)} · {timezone}</p>
    {loading ? <p role="status">Loading agenda…</p> : error ? <div role="alert"><p>{error}</p><button onClick={() => setRetry(retry + 1)}>Retry agenda</button></div> : result && <>
      {!result.items.length && <p role="status">No schedule occurrences in this period.</p>}
      {result.truncated && <p role="status">Results limited. Choose a shorter period to review more occurrences.</p>}
      <div className={`agenda-days agenda-${view.toLowerCase()}`}>{range.days.map(date => <section key={date} className={`agenda-day${date === dayInZone(timezone) ? " agenda-current" : ""}${date.slice(0, 7) !== day.slice(0, 7) ? " agenda-outside" : ""}`} aria-label={date}><h3><time dateTime={date}>{new Date(`${date}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", weekday: "short", day: "numeric", month: view === "Month" ? undefined : "short" })}</time></h3>{result.items.filter(item => item.day === date).map(item => <button className="agenda-occurrence" key={item.id} onClick={() => onSchedule(item.scheduled_task_id)}><time dateTime={item.at}>{new Date(item.at).toLocaleTimeString(undefined, { timeZone: timezone, hour: "2-digit", minute: "2-digit" })}</time><strong>{item.name}</strong><small>{item.kind === "execution" ? "Actual execution" : item.execution_status ? "Schedule occurrence" : "Projected occurrence"} · {item.schedule_status}</small><small>{item.execution_status ? `Execution: ${item.execution_status}` : "No execution recorded"}</small>{item.timezone !== timezone && <small>Schedule zone: {item.timezone}</small>}</button>)}</section>)}</div>
      {result.unavailable_sources.map(source => <p className="agenda-source-note" key={source}>{source}</p>)}
    </>}
    <footer><button onClick={onProjects}>Projects & Tasks</button>{onOpen && <button onClick={onOpen}>Detailed Agenda</button>}</footer>
  </section>;
}
