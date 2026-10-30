import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AtHandDrawer } from "./AtHandDrawer";
import { defaultDrawer } from "./atHandState";
const props = { timezone: "UTC", onSchedule: () => {}, onProjects: () => {}, state: { ...defaultDrawer(), open: true }, act: () => {}, member: "alice", connected: true, questionCount: 7, questionsMember: "alice", onAttention: () => {}, onAgenda: () => {}, toggleRef: { current: null }, saveFailed: false };
describe("At hand rendering boundaries", () => {
  it("never displays previous-member question counts or disconnected data", () => {
    expect(renderToStaticMarkup(<AtHandDrawer {...props} />)).toContain("7 waiting for your input");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} questionsMember="bob" />)).not.toContain("7 waiting");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} connected={false} />)).not.toContain("7 waiting");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} member="" />)).toContain("Loading member");
  });
  it("labels keyboard controls, renders expiration, and explains storage failure", () => {
    const html = renderToStaticMarkup(<AtHandDrawer {...props} state={{ ...props.state, widgets: [{ id: "trip", kind: "agenda", mode: "contextual", expiresAt: 1 }] }} saveFailed />);
    expect(html).toContain("Context expired");
    expect(html).toContain('aria-label="Pin Agenda"');
    expect(html).toContain('aria-label="Remove Agenda"');
    expect(html).toContain("Layout could not be saved");
  });
  it("keeps live activity samples member-scoped and unavailable when disconnected", () => {
    const state = { ...props.state, widgets: [{ id: "workload", kind: "workload", mode: "pinned" as const }, { id: "pdca", kind: "pdca", mode: "pinned" as const }] };
    const live = { ...props, state, activityMember: "alice", queueSamples: [{ at: Date.now(), ready: 7, processing: 2 }], pdcaSamples: [{ at: Date.now(), phase: "do" as const }] };
    const html = renderToStaticMarkup(<AtHandDrawer {...live} />);
    expect(html).toContain("9 total tasks");
    expect(html).toContain("7 waiting and 2 processing");
    expect(html).toContain("PDCA activity chart");
    for (const override of [{ activityMember: "bob" }, { connected: false }, { member: "bob" }]) {
      const unavailable = renderToStaticMarkup(<AtHandDrawer {...live} {...override} />);
      expect(unavailable).not.toContain("9 total tasks");
      expect(unavailable).not.toContain("PDCA activity chart");
    }
  });

});
