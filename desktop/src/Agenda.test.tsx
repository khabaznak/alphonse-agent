// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { Agenda, AgendaPreviewContent, AgendaTodayContent } from "./Agenda";
import { CALENDAR_CONFIGURATION_CHANGED, dayInZone, shiftDay, type AgendaResult, type CalendarSourceStatus } from "./agendaDates";
import { optionalManagement } from "./api";

const invoke = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke }));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const result: AgendaResult = { timezone: "America/Mexico_City", truncated: false, unavailable_sources: [], items: [
  { id: "occurrence", scheduled_task_id: "dinner", project_id: "", name: "Dinner planning", at: "2026-10-30T18:00:00Z", day: "2026-10-30", schedule_status: "paused", timezone: "America/Mexico_City", execution_status: null, run_id: null, kind: "occurrence" },
  { id: "execution", scheduled_task_id: "trip", project_id: "trip", name: "Trip reminder", at: "2026-10-31T18:00:00Z", day: "2026-10-31", schedule_status: "active", timezone: "UTC", execution_status: "failed", run_id: "run", kind: "execution" },
] };
const calendar: AgendaResult = { ...result, calendar_source_status: "connected", calendar_event_status: "complete", calendar_events: [
  { id: "long", title: "Passport pickup", start: "2026-10-30", end: "2026-11-01", all_day: true, end_unspecified: false },
  { id: "timed", title: "Coffee & snacks <script>", start: "2026-10-30T09:00:00-06:00", end: "2026-10-30T10:00:00-06:00", all_day: false, end_unspecified: false },
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
  it("displays Google Calendar events as noninteractive text before later scheduled work", () => {
    const div = document.createElement("div");
    div.innerHTML = render("Today", calendar);
    expect([...div.querySelectorAll(".agenda-preview-event strong")].map(x => x.textContent))
      .toEqual(["Passport pickup", "Coffee & snacks <script>", "Dinner planning"]);
    expect(div.querySelector("button.agenda-google-event")).toBeNull();
    expect(div.querySelector("script")).toBeNull();
    expect(div.querySelector(".agenda-google-event time")?.textContent).toBe("All day");
    expect(div.querySelectorAll(".agenda-google-event small")[1].textContent).toContain("Google Calendar");
  });
  it("marks both days of an all-day event and uses its exclusive end date", () => {
    const html = render("Month", calendar);
    expect(html).toContain('aria-label="2026-10-30: Google Calendar events and scheduled work"');
    expect(html).toContain('aria-label="2026-10-31: Google Calendar events and scheduled work"');
    expect(html).toContain('aria-label="2026-11-01"');
  });
});


describe("Today agenda", () => {
  it("renders accessible event rows with the available status and schedule target", () => {
    const html = renderToStaticMarkup(<AgendaTodayContent entries={result.items.map(item => ({ source: "schedule", item }))} day="2026-10-30" timezone="America/Mexico_City" onSchedule={() => {}} />);
    expect(html.match(/class="agenda-today-event"/g)).toHaveLength(2);
    expect(html).toContain('dateTime="2026-10-30T18:00:00Z"');
    expect(html).toContain("Dinner planning");
    expect(html).toContain("Scheduled · paused");
    expect(html).toContain("Actual execution · active · failed");
  });
  it("shows an empty state inside the list", () => {
    const html = renderToStaticMarkup(<AgendaTodayContent entries={[]} day="2026-10-30" timezone="UTC" onSchedule={() => {}} />);
    expect(html).toContain('class="agenda-today-list"');
    expect(html).toContain("No scheduled work on this day.");
  });
});

it("refreshes the open widget immediately after Calendar grant changes", async () => {
  let status: CalendarSourceStatus = "grant_required";
  invoke.mockImplementation(async (_command, args) => {
    if (args.method === "agenda") {
      expect(args.params.include_calendar_events).toBe(true);
      return { ...result, items: [], calendar_source_status: status };
    }
    if (args.params.action === "save_grant") status = "connected";
    if (args.params.action === "revoke_grant") status = "grant_required";
    return {};
  });
  const div = document.createElement("div");
  const root = createRoot(div);
  try {
    await act(async () => root.render(<Agenda member="owner" timezone="UTC" compact onSchedule={() => {}} onProjects={() => {}} />));
    expect(div.querySelector(".widget-source-details summary")?.textContent).toBe("Google Calendar needs calendar access");
    await act(async () => { await optionalManagement("save_grant", { operations: ["calendar_read"] }); });
    expect(div.querySelector(".widget-source-details summary")?.textContent).toBe("Google Calendar connected");
    expect(div.textContent).toContain("Shows events from your authorized Google calendar.");
    expect(div.textContent).toContain("No scheduled work.");
    await act(async () => { await optionalManagement("revoke_grant", {}); });
    expect(div.querySelector(".widget-source-details summary")?.textContent).toBe("Google Calendar needs calendar access");
  } finally {
    await act(async () => root.unmount());
    invoke.mockReset();
  }
});

it("keeps ordinary refreshes stable and clears private rows immediately on access changes", async () => {
  const day = dayInZone("UTC");
  const privateResult = { ...calendar, items: [], calendar_events: [
    { ...calendar.calendar_events![0], title: "Private appointment", start: day, end: shiftDay(day, 1) },
  ] };
  const unavailable: AgendaResult = { ...result, items: [], calendar_source_status: "connected", calendar_event_status: "unavailable" };
  let settlePoll: (value: AgendaResult) => void = () => {};
  let settleChanged: (value: AgendaResult) => void = () => {};
  invoke.mockResolvedValueOnce(privateResult)
    .mockImplementationOnce(() => new Promise<AgendaResult>(resolve => { settlePoll = resolve; }))
    .mockImplementationOnce(() => new Promise<AgendaResult>(resolve => { settleChanged = resolve; }));
  const div = document.createElement("div");
  const root = createRoot(div);
  try {
    await act(async () => root.render(<Agenda member="owner" timezone="UTC" compact onSchedule={() => {}} onProjects={() => {}} />));
    expect(div.textContent).toContain("Private appointment");
    await act(async () => window.dispatchEvent(new Event("focus")));
    expect(div.textContent).toContain("Private appointment");
    expect(div.textContent).not.toContain("Loading agenda");
    await act(async () => window.dispatchEvent(new Event(CALENDAR_CONFIGURATION_CHANGED)));
    expect(div.textContent).not.toContain("Private appointment");
    await act(async () => settlePoll(privateResult));
    expect(div.textContent).not.toContain("Private appointment");
    await act(async () => settleChanged(unavailable));
    expect(div.textContent).toContain("Google Calendar events unavailable");
    expect(div.textContent).not.toContain("No events or scheduled work.");
  } finally {
    await act(async () => root.unmount());
    invoke.mockReset();
  }
});

it("does not display a prior member's Calendar result while the new member loads", async () => {
  const day = dayInZone("UTC");
  const privateResult = { ...calendar, items: [], calendar_events: [
    { ...calendar.calendar_events![0], title: "Owner appointment", start: day, end: shiftDay(day, 1) },
  ] };
  invoke.mockImplementation((_command, args) => args.params.actor_user_id === "owner" ? Promise.resolve(privateResult) : new Promise(() => {}));
  const div = document.createElement("div");
  const root = createRoot(div);
  const element = (member: string) => <Agenda member={member} timezone="UTC" compact onSchedule={() => {}} onProjects={() => {}} />;
  try {
    await act(async () => root.render(element("owner")));
    expect(div.textContent).toContain("Owner appointment");
    await act(async () => root.render(element("member")));
    expect(div.textContent).not.toContain("Owner appointment");
    expect(div.textContent).toContain("Loading agenda");
  } finally {
    await act(async () => root.unmount());
    invoke.mockReset();
  }
});

it("shows provider events and coverage in Detailed Agenda without making them schedule controls", async () => {
  const day = dayInZone("UTC");
  const data: AgendaResult = { ...calendar, items: [], calendar_event_status: "partial", calendar_events: [
    { ...calendar.calendar_events![0], title: "Workshop", start: day, end: shiftDay(day, 1) },
  ] };
  invoke.mockResolvedValue(data);
  const div = document.createElement("div");
  const root = createRoot(div);
  const onSchedule = vi.fn();
  try {
    await act(async () => root.render(<Agenda member="owner" timezone="UTC" onSchedule={onSchedule} onProjects={() => {}} />));
    expect(div.querySelector(".agenda-day .agenda-google-event")?.textContent).toContain("Workshop");
    expect(div.textContent).toContain("Google Calendar results limited");
    const today = [...div.querySelectorAll(".agenda-view-switch button")].find(button => button.textContent === "Today");
    expect(today).toBeDefined();
    await act(async () => (today as HTMLButtonElement).click());
    const row = div.querySelector<HTMLDivElement>(".agenda-today-list .agenda-google-event");
    expect(row?.textContent).toContain("Workshop");
    await act(async () => row?.click());
    expect(onSchedule).not.toHaveBeenCalled();
  } finally {
    await act(async () => root.unmount());
    invoke.mockReset();
  }
});
