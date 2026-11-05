export type AgendaView = "Month" | "Week" | "Today";
export type AgendaItem = { id: string; scheduled_task_id: string; project_id: string; name: string; at: string; day: string; schedule_status: string; timezone: string; execution_status: string | null; run_id: string | null; kind: string };
export type AgendaResult = { items: AgendaItem[]; timezone: string; truncated: boolean; unavailable_sources: string[] };
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
