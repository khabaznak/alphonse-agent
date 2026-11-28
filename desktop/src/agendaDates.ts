export type AgendaView = "Month" | "Week" | "Today";
export type AgendaItem = { id: string; scheduled_task_id: string; project_id: string; name: string; at: string; day: string; schedule_status: string; timezone: string; execution_status: string | null; run_id: string | null; kind: string };
export const CALENDAR_CONFIGURATION_CHANGED = "alphonse:calendar-configuration-changed";
export type CalendarSourceStatus = "connected" | "not_connected" | "grant_required" | "selection_required" | "reconnect_required" | "unavailable";
export type CalendarEventStatus = "complete" | "partial" | "unavailable" | "api_disabled" | "approval_required" | "credential_busy";
export type GoogleCalendarEvent = { id: string; title: string; start: string; end: string; all_day: boolean; end_unspecified: boolean };
export type AgendaResult = {
  items: AgendaItem[]; timezone: string; truncated: boolean; unavailable_sources: string[];
  calendar_source_status?: CalendarSourceStatus; calendar_event_status?: CalendarEventStatus; calendar_events?: GoogleCalendarEvent[];
};
export type AgendaEntry = { source: "schedule"; item: AgendaItem } | { source: "google"; item: GoogleCalendarEvent };

/** Keep all-day dates on the calendar's civil dates and timed events on the owner's local dates. */
function occupiedDays(event: GoogleCalendarEvent, timezone: string): [string, string] | null {
  if (event.all_day) {
    const safeEnd = Number.isFinite(Date.parse(`${event.end}T12:00:00Z`));
    return safeEnd && event.start < event.end ? [event.start, shiftDay(event.end, -1)] : null;
  }
  const start = Date.parse(event.start), end = Date.parse(event.end);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  return [dayInZone(timezone, new Date(start)), dayInZone(timezone, new Date(end - 1))];
}

function entryOrder(entry: AgendaEntry) {
  if (entry.source === "schedule") return Date.parse(entry.item.at);
  return entry.item.all_day ? -Infinity : Date.parse(entry.item.start);
}

export function agendaEntriesByDay(result: AgendaResult, days: string[], timezone: string) {
  const entries = new Map(days.map(day => [day, [] as AgendaEntry[]]));
  for (const item of result.items) entries.get(item.day)?.push({ source: "schedule", item });
  if (result.calendar_source_status === "connected" &&
      (result.calendar_event_status === "complete" || result.calendar_event_status === "partial")) {
    for (const item of result.calendar_events ?? []) {
      const occupied = occupiedDays(item, timezone);
      if (!occupied) continue;
      const [first, last] = occupied;
      for (const day of days) if (first <= day && day <= last) entries.get(day)?.push({ source: "google", item });
    }
  }
  for (const dayEntries of entries.values()) dayEntries.sort((a, b) => entryOrder(a) - entryOrder(b));
  return entries;
}

export function dayInZone(timezone: string, now = new Date()) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-CA", { timeZone: timezone, year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(now).map(p => [p.type, p.value]));
  return `${parts.year}-${parts.month}-${parts.day}`;
}
export function shiftDay(day: string, amount: number) {
  const date = new Date(`${day}T12:00:00Z`); date.setUTCDate(date.getUTCDate() + amount); return date.toISOString().slice(0, 10);
}
export function agendaRange(day: string, view: AgendaView) {
  const date = new Date(`${day}T12:00:00Z`);
  let start = day, count = 1;
  if (view === "Week") { start = shiftDay(day, -(date.getUTCDay() + 6) % 7); count = 7; }
  if (view === "Month") { start = `${day.slice(0, 7)}-01`; start = shiftDay(start, -(new Date(`${start}T12:00:00Z`).getUTCDay() + 6) % 7); count = 42; }
  return { start, end: shiftDay(start, count), days: Array.from({ length: count }, (_, i) => shiftDay(start, i)) };
}
export function navigateAgenda(day: string, view: AgendaView, direction: number) {
  if (view !== "Month") return shiftDay(day, direction * (view === "Week" ? 7 : 1));
  const date = new Date(`${day.slice(0, 7)}-01T12:00:00Z`); date.setUTCMonth(date.getUTCMonth() + direction); return date.toISOString().slice(0, 10);
}
