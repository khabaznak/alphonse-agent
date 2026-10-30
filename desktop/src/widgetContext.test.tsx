import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AtHandDrawer } from "./AtHandDrawer";
import { changeDrawer, defaultDrawer, reconcileContext } from "./atHandState";
import { authorizedWidgets, suggestedPlacements, type SkillWidget, type WidgetContext } from "./widgetContext";
const calendar: SkillWidget = { id: `skill-widget:${"a".repeat(32)}`, kind: "skill", member: "alice", skillId: "skill:calendar", contextId: "dinner-trip", scope: "member", title: "Calendar", binding: "agenda", status: "ready", text: "Review scheduled work and its history.", relevant: true, suggestion: "show", expiresAt: Date.now() + 100000 };
const exercise: SkillWidget = { ...calendar, id: `skill-widget:${"b".repeat(32)}`, title: "Exercise progress", binding: "memory", historical: true, text: "Past session: three sets", contextId: "past-session" };
const snapshot = (widgets: SkillWidget[]): WidgetContext => ({ version: 1, member: "alice", widgets });
const props = { act: () => {}, member: "alice", connected: true, questionCount: 0, questionsMember: "alice", onAttention: () => {}, onAgenda: () => {}, toggleRef: { current: null }, saveFailed: false };
describe("authorized contextual widgets", () => {
  it("resurfaces dinner/trip calendar and past-session progress, preserving unrelated locks", () => {
    let state = changeDrawer(defaultDrawer(), { type: "mode", id: "questions", mode: "locked" });
    const locked = state.widgets[0];
    const rows = authorizedWidgets(snapshot([calendar, exercise]), "alice");
    state = reconcileContext(state, suggestedPlacements(rows), Date.now());
    const again = reconcileContext(state, suggestedPlacements(rows), Date.now());
    expect(again).toBe(state);
    expect(state.widgets[0]).toBe(locked);
    expect(state.open).toBe(false);
    const html = renderToStaticMarkup(<AtHandDrawer {...props} state={{ ...state, open: true }} widgetContext={snapshot(rows)} />);
    expect(html).toContain("Calendar");
    expect(html).toContain("Past session: three sets");
    expect(html).toContain("Historical context");
    state = changeDrawer(state, { type: "remove", id: calendar.id });
    expect(reconcileContext(state, suggestedPlacements(rows), Date.now())).toBe(state);
  });
  it("suppresses revoked/missing/unavailable data even on a locked card and member changes", () => {
    let state = reconcileContext(defaultDrawer(), [exercise], Date.now());
    state = changeDrawer(state, { type: "mode", id: exercise.id, mode: "locked" });
    for (const context of [undefined, snapshot([]), snapshot([{ ...exercise, status: "unavailable", text: "" }]), { ...snapshot([exercise]), member: "bob" }]) {
      const html = renderToStaticMarkup(<AtHandDrawer {...props} state={{ ...state, open: true }} widgetContext={context} />);
      expect(html).not.toContain("Past session: three sets");
      expect(html).toContain("User locked");
    }
    const disconnected = renderToStaticMarkup(<AtHandDrawer {...props} connected={false} state={state} widgetContext={snapshot([exercise])} />);
    expect(disconnected).not.toContain("Past session: three sets");
  });
  it("treats lifecycle proposals as suggestions and renders untrusted text without executing it", () => {
    const rows = ["pin", "remove", "expire", "update"].map((suggestion) => ({ ...exercise, suggestion } as SkillWidget));
    expect(suggestedPlacements(rows).map((row) => row.suggestion)).toEqual(["pin"]);
    const state = reconcileContext(defaultDrawer(), [exercise], Date.now());
    const html = renderToStaticMarkup(<AtHandDrawer {...props} state={state} widgetContext={snapshot([{ ...exercise, suggestion: "remove", text: "<script>alert(1)</script>" }])} />);
    expect(html).toContain("Alphonse suggests");
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("<script>");
    expect(authorizedWidgets(snapshot([{ ...exercise, binding: "javascript" } as unknown as SkillWidget]), "alice")).toEqual([]);
  });
});
