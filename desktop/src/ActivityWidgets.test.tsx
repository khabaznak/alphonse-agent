import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ActivityWidget, timerFrame, validActivity, type ActivityData, type TimerRecord } from "./ActivityWidgets";
import { AtHandDrawer } from "./AtHandDrawer";
import { changeDrawer, defaultDrawer, reconcileContext } from "./atHandState";
import { authorizedWidgets, type SkillWidget } from "./widgetContext";

const start = Date.parse("2026-10-29T12:00:00Z");
const timer: TimerRecord = { id: "interval", title: "Interval", scope: "session", recorded_at: new Date(start).toISOString(), status: "running", elapsed_ms: 10000, duration_ms: 60000, running_since: new Date(start).toISOString() };
const data: ActivityData = { family: "timer", artifactId: "artifact.fixture", owner: "alice", source: "Exercise fixture", updatedAt: new Date(start).toISOString(), expiresAt: start + 3600000, items: [timer] };
const html = (value: ActivityData, now = start) => renderToStaticMarkup(<ActivityWidget data={value} now={now} timezone="UTC" onRequest={() => {}} />);
const row: SkillWidget = { id: `skill-widget:${"c".repeat(32)}`, kind: "skill", member: "alice", skillId: "skill:exercise", contextId: "yesterday", scope: "member", title: "Exercise", binding: "activity", activity: data, status: "ready", text: "", relevant: true, suggestion: "show", expiresAt: start + 3600000 };

describe("source-authoritative activity widgets", () => {
  it("derives time across rerenders, duplicate polls, restarts and reconnects without counters", () => {
    expect(timerFrame(timer, start + 5000).remaining).toBe(45000);
    const restored = JSON.parse(JSON.stringify(timer));
    expect(timerFrame(restored, start + 40000).remaining).toBe(10000);
    expect(timerFrame(timer, start + 40000)).toEqual(timerFrame(restored, start + 40000));
    expect(timerFrame({ ...timer, status: "paused", running_since: null, elapsed_ms: 25000 }, start + 900000).remaining).toBe(35000);
    const resumed = { ...timer, elapsed_ms: 25000, running_since: new Date(start + 100000).toISOString() };
    expect(timerFrame(resumed, start + 110000).remaining).toBe(25000);
    expect(timerFrame({ ...timer, status: "cancelled", running_since: null }, start + 900000).elapsed).toBe(10000);
    expect(timerFrame(timer, start - 5000).elapsed).toBe(10000);
  });
  it("never turns clock expiry or target attainment into verified completion", () => {
    const expired = html(data, start + 61000);
    expect(expired).toContain("Time elapsed · completion unconfirmed");
    expect(expired).not.toContain("Source reports completed");
    expect(expired).toContain('role="timer"');
    expect(html({ ...data, items: [{ ...timer, status: "completed", running_since: null, elapsed_ms: 45000 }] })).toContain("Source reports completed");
  });
  it("labels yesterday's recorded counts separately from project and illustrative progress", () => {
    const progress: ActivityData = { ...data, family: "tracker", items: [
      { id: "yesterday", title: "Yesterday’s exercise", scope: "session", recorded_at: "2026-10-28T12:00:00Z", status: "active", value: 3, target: 5, unit: "sets", display: "dots" },
      { id: "project", title: "Training plan", scope: "project", recorded_at: data.updatedAt, status: "active", value: 2, target: 10, unit: "sessions", display: "bar" },
      { id: "sample", title: "Example", scope: "sample", recorded_at: data.updatedAt, status: "active", value: 75, target: 100, unit: "percent", display: "bar" },
    ] };
    const result = html(progress);
    expect(result).toContain("3 of 5 sets");
    expect(result).toContain("Recorded session");
    expect(result).toContain("Project progress");
    expect(result).toContain("Illustrative sample · not recorded progress");
    expect(result).toContain('aria-valuetext="3 of 5 sets"');
    expect(result).toContain('data-filled="true"');
    expect(html({ ...progress, items: [{ ...progress.items[0], value: 4 } as typeof progress.items[0]] })).toContain("4 of 5 sets");
  });
  it("keeps locks and strips data on disconnect, revoked access, expiry and actor changes", () => {
    const live = { ...row, expiresAt: Date.now() + 100000, activity: { ...data, expiresAt: Date.now() + 100000 } };
    let state = reconcileContext(defaultDrawer(), [live], Date.now());
    state = changeDrawer(state, { type: "mode", id: row.id, mode: "locked" });
    const props = { state, act: () => {}, member: "alice", connected: true, questionCount: 0, questionsMember: "alice", onAttention: () => {}, onAgenda: () => {}, onSchedule: () => {}, onProjects: () => {}, timezone: "UTC", toggleRef: { current: null }, saveFailed: false };
    const context = { version: 1, member: "alice", widgets: [live] };
    expect(reconcileContext(state, [live], Date.now())).toBe(state);
    expect(renderToStaticMarkup(<AtHandDrawer {...props} widgetContext={context} />)).toContain("Exercise fixture");
    for (const override of [{ connected: false }, { widgetContext: { ...context, widgets: [] } }, { member: "bob" }]) {
      const result = renderToStaticMarkup(<AtHandDrawer {...props} widgetContext={context} {...override} />);
      expect(result).not.toContain("Exercise fixture");
      expect(result).toContain("User locked");
    }
    expect(html(data, data.expiresAt)).toContain("Unavailable");
    expect(authorizedWidgets({ ...context, widgets: [{ ...live, activity: { ...data, items: [{ ...timer, duration_ms: 0 }] } }] }, "alice")).toEqual([]);
  });
  it("rejects malformed quantities and clocks", () => {
    for (const item of [{ ...timer, elapsed_ms: NaN }, { ...timer, running_since: null }, { ...timer, status: "paused" }, { ...timer, running_since: "2026-10-29T12:00:00" }]) {
      expect(validActivity({ ...data, items: [item as TimerRecord] })).toBe(false);
    }
  });
});
