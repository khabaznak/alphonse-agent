import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, it, expect } from "vitest";
import { CareWidget, careDraft, validCare, type CareData } from "./CareWidget";
import { AtHandDrawer } from "./AtHandDrawer";
import { authorizedWidgets, type SkillWidget } from "./widgetContext";
import { changeDrawer, defaultDrawer, saveDrawer } from "./atHandState";
const data: CareData = { member: "alice", source: "Synthetic Doctor", recordedAt: new Date().toISOString(), expiresAt: Date.now() + 60000, items: [{ id: "routine-1", definition: "<img src=x> Synthetic routine", adherence: "unknown", dueAt: null, recordedAt: null, writeState: "none" }] };
const widget: SkillWidget = { id: `skill-widget:${"c".repeat(32)}`, kind: "skill", member: "alice", skillId: "skill:doctor", contextId: "care", scope: "member", title: "Care", binding: "care", status: "ready", text: "", relevant: true, suggestion: "show", expiresAt: Date.now() + 60000, care: data };
const context = { version: 1, member: "alice", widgets: [widget] };
const render = (payload = data) => renderToStaticMarkup(<CareWidget data={payload} member="alice" onRequest={() => {}} timezone="America/Mexico_City" />);
describe("Treatment and care projections", () => {
  it("shows provenance, unknown adherence, escaped definitions and explicit workflow actions", () => {
    const html = render();
    for (const text of ["Member: alice", "Synthetic Doctor", "Source recorded", "routine-1", "Adherence unknown", "Due time unknown", "Request logging", "Request correction", "Open record / ask"]) expect(html).toContain(text);
    expect(html).toContain("&lt;img"); expect(html).not.toContain("<img"); expect(html).not.toContain("Explicitly recorded as missed");
    const draft = careDraft(data, data.items[0], "Correct record");
    expect(draft).toContain("check current access"); expect(draft).toContain("verifying the persisted effect"); expect(draft).toContain("Ask for missing details");
  });
  it("distinguishes source-confirmed writes, failures, unconfirmed writes and explicit missed records", () => {
    for (const [writeState, message] of [["failed", "Write failed"], ["unconfirmed", "Write unconfirmed"], ["verified", "Source confirms a verified write"]] as const) {
      expect(render({ ...data, items: [{ ...data.items[0], writeState }] })).toContain(message);
    }
    expect(render({ ...data, items: [{ ...data.items[0], adherence: "missed", recordedAt: data.recordedAt, dueAt: data.recordedAt }] })).toContain("Explicitly recorded as missed");
    expect(render({ ...data, items: [] })).toContain("Adherence is unknown");
  });
  it("discards revoked, expired, malformed and cross-member payloads", () => {
    expect(validCare({ ...data, items: [{ ...data.items[0], adherence: "missed" }] }, "alice")).toBe(false);
    expect(validCare({ ...data, items: [...data.items, ...data.items] }, "alice")).toBe(false);
    expect(authorizedWidgets(context, "bob")).toEqual([]);
    expect(authorizedWidgets({ ...context, widgets: [{ ...widget, care: { ...data, member: "bob" } }] }, "alice")).toEqual([]);
    expect(render({ ...data, expiresAt: 1 })).toContain("Care records unavailable");
    expect(renderToStaticMarkup(<CareWidget member="alice" />)).toContain("Missing data does not mean missed care");
  });
  it("catalog and contextual cards respect disconnect/member changes and persist only layout", () => {
    const state = { ...changeDrawer(defaultDrawer(), { type: "add", kind: "care" }), open: true };
    const props = { state, act: () => {}, member: "alice", connected: true, questionCount: 0, questionsMember: "alice", onAttention: () => {}, onAgenda: () => {}, toggleRef: { current: null }, saveFailed: false, widgetContext: context };
    expect(renderToStaticMarkup(<AtHandDrawer {...props} />)).toContain("Synthetic Doctor");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} connected={false} />)).not.toContain("Synthetic Doctor");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} member="bob" />)).not.toContain("Synthetic Doctor");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} widgetContext={{ ...context, widgets: [{ ...widget, status: "unavailable", care: undefined }] }} />)).toContain("Care records unavailable");
    let stored = ""; saveDrawer({ setItem: (_key, value) => { stored = value; } }, "alice", state);
    expect(stored).not.toContain("Synthetic Doctor"); expect(stored).not.toContain("routine-1");
  });
});
