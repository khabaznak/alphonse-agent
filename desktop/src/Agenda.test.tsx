import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AgendaPreviewContent } from "./Agenda";
import type { AgendaResult } from "./agendaDates";

const result: AgendaResult = { timezone: "America/Mexico_City", truncated: false, unavailable_sources: [], items: [
  { id: "occurrence", scheduled_task_id: "dinner", project_id: "", name: "Dinner planning", at: "2026-10-30T18:00:00Z", day: "2026-10-30", schedule_status: "paused", timezone: "America/Mexico_City", execution_status: null, run_id: null, kind: "occurrence" },
  { id: "execution", scheduled_task_id: "trip", project_id: "trip", name: "Trip reminder", at: "2026-10-31T18:00:00Z", day: "2026-10-31", schedule_status: "active", timezone: "UTC", execution_status: "failed", run_id: "run", kind: "execution" },
] };
const render = (view: "Month" | "Week" | "Today", data = result) => renderToStaticMarkup(<AgendaPreviewContent result={data} day="2026-10-30" view={view} timezone="America/Mexico_City" onDay={() => {}} onSchedule={() => {}} />);

describe("compact Agenda content", () => {
  it("shows a seven-column month picker and only the selected day's event details", () => {
    const html = render("Month");
    expect(html).toContain('aria-label="Month dates"');
    expect(html.match(/aria-pressed=/g)).toHaveLength(42);
    expect(html).toContain('aria-label="2026-10-31: scheduled work"');
    expect(html).toContain("Dinner planning");
    expect(html).not.toContain("Trip reminder");
    expect(html).not.toContain('class="agenda-day');
  });
  it("keeps all seven week rows and actual execution failures distinct from planned work", () => {
    const html = render("Week");
    expect(html.match(/<section /g)).toHaveLength(7);
    expect(html).toContain("Dinner planning");
    expect(html).toContain("paused · not run");
    expect(html).toContain("Actual execution · active · failed");
    expect(html).toContain("UTC");
  });
  it("keeps empty Today honest and does not invent calendar entries", () => {
    const html = render("Today", { ...result, items: [] });
    expect(html).toContain("No scheduled work.");
    expect(html).not.toContain("Dinner planning");
    expect(html).not.toContain("Trip reminder");
  });
});
