import React from "react";
import { renderToStaticMarkup as render } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { HomeSetup, HomeWidget, sensorState, validHome, type HomeData, type Sensor } from "./HomeWidgets";
import { authorizedWidgets, type SkillWidget } from "./widgetContext";
import { AtHandDrawer } from "./AtHandDrawer";
import { changeDrawer, defaultDrawer, reconcileContext, saveDrawer } from "./atHandState";
const now = Date.now();
const sensor: Sensor = { id: "temp", name: "Fixture <room>", source: "Fixture provider", unit: "°C", value: 23.5, observed_at: new Date(now - 1000).toISOString(), stale_after_seconds: 60, state: "connected" };
const source = { artifactId: "artifact.fixture", owner: "alice", source: "Fixture source", updatedAt: new Date(now).toISOString(), expiresAt: now + 60000 };
const sensors: HomeData = { ...source, family: "sensors", items: [sensor] };
const contacts: HomeData = { ...source, family: "emergency_contacts", items: [{ id: "contact", name: "Fixture friend", role: "Chosen contact", phone: "+1 555 0100", email: "fixture@example.invalid" }] };
const widget: SkillWidget = { id: `skill-widget:${"a".repeat(32)}`, kind: "skill", member: "alice", scope: "member", skillId: "skill:fixture", contextId: "fixture", title: "Home sensors", binding: "home", home: sensors, status: "ready", text: "", relevant: true, suggestion: "show", expiresAt: now + 60000 };
describe("Home widgets", () => {
  it("renders units, source and timestamps as escaped text", () => {
    const html = render(<HomeWidget data={sensors} timezone="America/Mexico_City" />);
    expect(html).toContain("23.5 °C"); expect(html).toContain("Current reading");
    expect(html).toContain("Fixture provider"); expect(html).toContain("Last reading:");
    expect(html).toContain("Fixture &lt;room&gt;"); expect(html).not.toContain("<room>");
  });
  it("ages readings independently of locks or poll refresh and preserves offline/error labels", () => {
    expect(sensorState(sensor, source.expiresAt, now + 59000)).toBe("stale");
    expect(sensorState(sensor, now - 1, now)).toBe("stale");
    for (const state of ["stale", "disconnected", "error"] as const) {
      expect(sensorState({ ...sensor, state }, source.expiresAt, now)).toBe(state);
      const html = render(<HomeWidget data={{ ...sensors, items: [{ ...sensor, state }] }} />);
      expect(html).toContain("not live"); expect(html).toContain("Last known: "); expect(html).not.toContain("Current reading");
    }
    expect(render(<HomeWidget data={{ ...sensors, updatedAt: new Date(now - 120000).toISOString(), expiresAt: now - 1, items: [{ ...sensor, observed_at: new Date(now - 120000).toISOString() }] }} />)).toContain("Stale source");
    expect(render(<HomeWidget data={{ ...sensors, items: [{ ...sensor, value: null, observed_at: null, state: "disconnected" }] }} />)).toContain("Never received");
  });
  it("only exposes configured manual contact actions and no invented fallback numbers", () => {
    const html = render(<HomeWidget data={contacts} onRequest={() => {}} />);
    expect(html).toContain("Fixture friend"); expect(html).toContain("Chosen contact");
    expect(html).toContain("Copy phone"); expect(html).toContain("Copy email"); expect(html).not.toContain("tel:");
    expect(html).toContain("Review / edit with Alphonse");
    expect(render(<HomeWidget data={{ ...contacts, items: [] }} />)).toContain("No emergency contacts configured");
    expect(render(<HomeWidget data={{ ...contacts, items: [{ ...contacts.items[0], phone: null, email: null }] }} />)).toContain("No contact action configured");
    expect(render(<HomeWidget data={{ ...contacts, expiresAt: now - 1 }} />)).not.toContain("Fixture friend");
    expect(render(<HomeSetup family="emergency_contacts" onRequest={() => {}} />)).toContain("Set up with Alphonse");
  });
  it("rejects malformed payloads and never trusts saved identities as permission", () => {
    expect(validHome({ ...sensors, items: [sensor, sensor] })).toBe(false);
    expect(validHome({ ...sensors, items: [{ ...sensor, value: NaN }] })).toBe(false);
    expect(validHome({ ...sensors, items: [{ ...sensor, observed_at: null }] })).toBe(false);
    expect(validHome({ ...contacts, items: [{ ...contacts.items[0], phone: "javascript:alert(1)" }] })).toBe(false);
    const context = { version: 1, member: "alice", widgets: [widget] };
    expect(authorizedWidgets(context, "bob")).toEqual([]);
    expect(authorizedWidgets({ ...context, widgets: [{ ...widget, home: undefined }] }, "alice")).toEqual([]);
    let state = changeDrawer(defaultDrawer(), { type: "add", kind: "sensors" });
    state = changeDrawer(state, { type: "mode", id: "sensors", mode: "locked" });
    state = reconcileContext(state, [widget, widget], now);
    expect(state.widgets.filter((row) => row.id === widget.id)).toHaveLength(1);
    expect(state.widgets.find((row) => row.id === "sensors")?.mode).toBe("locked");
    const props = { state: { ...state, open: true }, act: () => {}, member: "alice", connected: true, questionCount: 0, questionsMember: "alice", onAttention: () => {}, onAgenda: () => {}, toggleRef: { current: null }, saveFailed: false, widgetContext: context };
    expect(render(<AtHandDrawer {...props} />)).toContain("23.5");
    for (const changed of [{ connected: false }, { member: "bob" }, { widgetContext: { ...context, widgets: [] } }]) expect(render(<AtHandDrawer {...props} {...changed} />)).not.toContain("23.5");
    let stored = ""; saveDrawer({ setItem: (_, value) => { stored = value; } }, "alice", state);
    expect(stored).not.toContain("23.5"); expect(stored).not.toContain("Fixture provider");
  });
});
