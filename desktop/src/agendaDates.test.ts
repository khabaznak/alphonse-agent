import { describe, expect, it } from "vitest";
import { agendaRange, dayInZone, navigateAgenda } from "./agendaDates";
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
});
