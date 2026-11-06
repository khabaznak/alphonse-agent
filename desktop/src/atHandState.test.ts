import { describe, expect, it } from "vitest";
import { changeDrawer, defaultDrawer, drawerStorageKey, readDrawer, reconcileContext, saveDrawer, widgetAvailability, type DrawerState } from "./atHandState";
const temporary = { id: "context:trip", kind: "agenda", expiresAt: 2000 };
describe("At hand layout invariants", () => {
  it("round trips member visibility, identities, modes, order and dismissal without payloads", () => {
    const values = new Map<string, string>();
    const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
    let state = changeDrawer(defaultDrawer(), { type: "move", id: "agenda", direction: -1 });
    state = changeDrawer(state, { type: "mode", id: "agenda", mode: "locked" });
    state = changeDrawer(state, { type: "visibility", open: true });
    state = changeDrawer(state, { type: "remove", id: "questions" });
    expect(saveDrawer(storage, "alice", state)).toBe(true);
    expect(readDrawer(storage, "alice")).toEqual(state);
    expect(readDrawer(storage, "bob")).toEqual(defaultDrawer());
    expect(saveDrawer(storage, "", state)).toBe(false);
    expect(values.size).toBe(1);
    // Unknown fields from older/malformed snapshots are never retained as content.
    values.set(drawerStorageKey("bob"), JSON.stringify({ ...state, widgets: [{ ...state.widgets[0], payload: "private", title: "secret" }] }));
    expect(JSON.stringify(readDrawer(storage, "bob"))).not.toMatch(/private|secret|payload/);
  });
  it("enforces locked removal/movement and blocks crossing a lock until explicit unlock", () => {
    const state = changeDrawer(defaultDrawer(), { type: "mode", id: "questions", mode: "locked" });
    expect(changeDrawer(state, { type: "remove", id: "questions" })).toBe(state);
    expect(changeDrawer(state, { type: "move", id: "questions", direction: 1 })).toBe(state);
    expect(changeDrawer(state, { type: "move", id: "agenda", direction: -1 })).toBe(state);
    const unlocked = changeDrawer(state, { type: "mode", id: "questions", mode: "pinned" });
    expect(changeDrawer(unlocked, { type: "remove", id: "questions" }).widgets).toHaveLength(1);
  });
  it("automatic reconnects cannot duplicate, reorder, downgrade locks or resurrect dismissed widgets", () => {
    let state = changeDrawer(defaultDrawer(), { type: "mode", id: "agenda", mode: "locked" });
    state = reconcileContext(state, [temporary, temporary], 1000);
    const locked = state.widgets[1];
    expect(reconcileContext(state, [temporary, { ...temporary, id: "agenda" }], 1000)).toBe(state);
    expect(reconcileContext(state, [], 3000)).toBe(state);
    expect(state.widgets[1]).toBe(locked);
    expect(state.widgets).toHaveLength(3);
    const removed = changeDrawer(state, { type: "remove", id: temporary.id });
    expect(reconcileContext(removed, [temporary], 1000)).toBe(removed);
    expect(changeDrawer(removed, { type: "add", kind: "agenda" })).toBe(removed);
  });
  it("expires only temporary cards and never edits underlying work", () => {
    const state = reconcileContext(defaultDrawer(), [temporary], 1000);
    const contextual = state.widgets[2];
    expect(widgetAvailability(contextual, "alice", true, 2001)).toBe("expired");
    for (const mode of ["pinned", "locked"] as const) expect(widgetAvailability({ ...contextual, mode }, "alice", true, 2001)).toBe("ready");
    expect(widgetAvailability(contextual, "", true, 1000)).toBe("loading");
    expect(widgetAvailability(contextual, "alice", false, 1000)).toBe("unavailable");
    expect(widgetAvailability({ ...contextual, kind: "missing" }, "alice", true, 1000)).toBe("unavailable");
    expect(reconcileContext(state, [{ ...temporary, id: "old", expiresAt: 1 }], 1000)).toBe(state);
    // Pure reducers have no daemon capability: dismissal only touches layout metadata.
    expect(changeDrawer(state, { type: "remove", id: contextual.id }).dismissed).toEqual([contextual.id]);
  });
  it("recovers malformed storage, deduplicates saved IDs and reports failed persistence", () => {
    expect(readDrawer({ getItem: () => "{" }, "alice")).toEqual(defaultDrawer());
    expect(readDrawer({ getItem: () => { throw Error(); } }, "alice")).toEqual(defaultDrawer());
    const state: DrawerState = { ...defaultDrawer(), widgets: [defaultDrawer().widgets[0], defaultDrawer().widgets[0]] };
    expect(readDrawer({ getItem: () => JSON.stringify(state) }, "alice").widgets).toHaveLength(1);
    expect(saveDrawer({ setItem: () => { throw Error(); } }, "alice", state)).toBe(false);
    expect(changeDrawer(defaultDrawer(), { type: "add", kind: "unknown" })).toEqual(defaultDrawer());
  });
  it("treats Workload and PDCA as optional persisted widgets with the usual locks", () => {
    expect(defaultDrawer().widgets.some(widget => ["workload", "pdca"].includes(widget.kind))).toBe(false);
    let state = changeDrawer(defaultDrawer(), { type: "add", kind: "workload" });
    state = changeDrawer(state, { type: "add", kind: "pdca" });
    state = changeDrawer(state, { type: "mode", id: "workload", mode: "locked" });
    expect(changeDrawer(state, { type: "remove", id: "workload" })).toBe(state);
    expect(changeDrawer(state, { type: "add", kind: "pdca" })).toBe(state);
    expect(readDrawer({ getItem: () => JSON.stringify(state) }, "alice")).toEqual(state);
    const removed = changeDrawer(state, { type: "remove", id: "pdca" });
    expect(removed.widgets.some(widget => widget.id === "pdca")).toBe(false);
    expect(removed.widgets.find(widget => widget.id === "workload")?.mode).toBe("locked");
  });

});
