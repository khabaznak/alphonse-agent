import { describe, expect, it } from "vitest";
import { agendaEntriesByDay, agendaRange, dayInZone, navigateAgenda, type AgendaResult } from "./agendaDates";
describe("Agenda civil dates", () => {
  it("spans month and year boundaries with Monday weeks", () => {
    expect(agendaRange("2027-01-01", "Week")).toMatchObject({ start: "2026-12-28", end: "2027-01-04" });
    const month = agendaRange("2026-02-15", "Month");
    expect(month.start).toBe("2026-01-26"); expect(month.end).toBe("2026-03-09"); expect(month.days).toHaveLength(42);
  });
  it("navigates leap years and short months without overflow", () => {
    expect(navigateAgenda("2028-01-31", "Month", 1)).toBe("2028-02-01");
    expect(navigateAgenda("2028-02-28", "Today", 1)).toBe("2028-02-29");
    expect(navigateAgenda("2026-11-01", "Week", 1)).toBe("2026-11-08");
  });
  it("returns today in the configured zone on either side of midnight and DST", () => {
    const now = new Date("2026-11-01T05:30:00Z");
    expect(dayInZone("America/Los_Angeles", now)).toBe("2026-10-31");
    expect(dayInZone("Asia/Tokyo", now)).toBe("2026-11-01");
    expect(agendaRange("2026-11-01", "Today").end).toBe("2026-11-02");
  });
  it("projects timed overlap in the viewer's zone without a phantom day at midnight", () => {
    const result: AgendaResult = { items: [], timezone: "America/Mexico_City", truncated: false, unavailable_sources: [],
      calendar_source_status: "connected", calendar_event_status: "partial", calendar_events: [
        { id: "ends-at-midnight", title: "Late workshop", start: "2026-10-30T03:30:00Z", end: "2026-10-30T06:00:00Z", all_day: false, end_unspecified: false },
        { id: "spans-midnight", title: "Overnight class", start: "2026-10-30T04:00:00Z", end: "2026-10-30T08:00:00Z", all_day: false, end_unspecified: false },
      ] };
    const entries = agendaEntriesByDay(result, ["2026-10-29", "2026-10-30"], result.timezone);
    expect(entries.get("2026-10-29")?.map(entry => entry.item.id)).toEqual(["ends-at-midnight", "spans-midnight"]);
    expect(entries.get("2026-10-30")?.map(entry => entry.item.id)).toEqual(["spans-midnight"]);
    expect(agendaEntriesByDay({ ...result, calendar_source_status: "grant_required" }, ["2026-10-29"], result.timezone).get("2026-10-29")).toEqual([]);
  });
});
