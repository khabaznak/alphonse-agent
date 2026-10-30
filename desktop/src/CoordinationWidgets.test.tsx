import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { appendWidgetDraft, CoordinationList, NativeCoordinationBody, validCoordination, widgetTime, workflowDraft, type CoordinationData, type NativeCoordination } from "./CoordinationWidgets";
import { AtHandDrawer } from "./AtHandDrawer";
import { authorizedWidgets, type SkillWidget } from "./widgetContext";
import { changeDrawer, defaultDrawer, reconcileContext } from "./atHandState";
const data: CoordinationData = { family: "shopping", artifactId: "artifact.fixture", owner: "alice", source: "Fixture list", updatedAt: new Date().toISOString(), expiresAt: Date.now() + 60000, items: [{ id: "milk", title: "<img src=x onerror=alert(1)>", status: "pending" }] };
const routes = { onAgenda: () => {}, onWork: () => {}, onRequest: () => {}, onItem: () => {} };
const widget: SkillWidget = { id: `skill-widget:${"a".repeat(32)}`, kind: "skill", member: "alice", skillId: "skill:fixture", contextId: "planning", scope: "member", title: "Shopping", binding: "coordination", status: "ready", text: "", relevant: true, suggestion: "show", expiresAt: Date.now() + 60000, coordination: data };
const context = { version: 1, member: "alice", widgets: [widget] };
describe("Coordination widgets", () => {
  it("renders source, item identity and workflow actions as text, never executable UI", () => {
    const html = renderToStaticMarkup(<CoordinationList data={data} {...routes} />);
    expect(html).toContain("Fixture list"); expect(html).toContain("Owner: alice"); expect(html).toContain("Item: milk");
    expect(html).toContain("&lt;img"); expect(html).not.toContain("<img");
    expect(html).toContain("Request completion"); expect(html).toContain("Request edit"); expect(html).toContain("Open in Projects &amp; Tasks");
    const draft = workflowDraft(data, data.items[0], "Complete this item");
    expect(draft).toContain("verify the effect");
    expect(appendWidgetDraft(appendWidgetDraft("Existing draft", draft), draft)).toBe(`Existing draft\n\n${draft}`);
  });
  it("never converts missing, expired, malformed or revoked data into an empty success", () => {
    expect(validCoordination({ ...data, items: [data.items[0], data.items[0]] })).toBe(false);
    expect(validCoordination({ ...data, items: [{ ...data.items[0], when: "not a date" }] })).toBe(false);
    expect(renderToStaticMarkup(<CoordinationList data={{ ...data, expiresAt: 1 }} {...routes} />)).toContain("Unavailable");
    expect(renderToStaticMarkup(<CoordinationList data={{ ...data, items: [] }} {...routes} />)).toContain("No items");
    expect(authorizedWidgets(context, "bob")).toEqual([]);
    expect(authorizedWidgets({ ...context, widgets: [{ ...widget, coordination: undefined }] }, "alice")).toEqual([]);
  });
  it("resurfaces without duplicating or moving locks, and hides disconnected payloads", () => {
    let state = changeDrawer(defaultDrawer(), { type: "mode", id: "agenda", mode: "locked" });
    state = reconcileContext(state, [widget, widget], Date.now());
    expect(state.widgets.filter((row) => row.id === widget.id)).toHaveLength(1);
    expect(state.widgets.find((row) => row.id === "agenda")?.mode).toBe("locked");
    const props = { ...routes, state: { ...state, open: true }, act: () => {}, member: "alice", connected: true, questionCount: 0, questionsMember: "alice", onAttention: () => {}, toggleRef: { current: null }, saveFailed: false, widgetContext: context };
    expect(renderToStaticMarkup(<AtHandDrawer {...props} />)).toContain("Fixture list");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} connected={false} />)).not.toContain("Owner: alice");
  });
  it("distinguishes native loading, unavailable and empty states and displays save failure", () => {
    const state: NativeCoordination = { member: "alice", loading: false, error: "", work: [], reminders: [] };
    const controller = { state, notice: "Save failed or unconfirmed", busy: "", change: async () => {}, refresh: async () => {} };
    expect(renderToStaticMarkup(<NativeCoordinationBody family="reminders" controller={controller} {...routes} />)).toContain("Save failed");
    expect(renderToStaticMarkup(<NativeCoordinationBody family="todos" controller={controller} {...routes} />)).toContain("No active to-dos");
    expect(renderToStaticMarkup(<NativeCoordinationBody family="todos" controller={{ ...controller, state: { ...state, loading: true } }} {...routes} />)).toContain("Loading");
    expect(renderToStaticMarkup(<NativeCoordinationBody family="todos" controller={{ ...controller, state: { ...state, error: "Tasks unavailable." } }} {...routes} />)).toContain("Unavailable");
    expect(widgetTime("2026-11-02T15:00:00+00:00", "America/Mexico_City")).toContain("9:00");
  });
});
