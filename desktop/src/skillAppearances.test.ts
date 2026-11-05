import { describe, expect, it } from "vitest";
import { appearanceAllowed, BASE_CONTEXT, BASE_LATCH, latchAppearance, readAppearancePreferences, saveAppearancePreferences, type AppearanceContext } from "./skillAppearances";

const context = (skill: string | null, task = "task-1", revision = "v1"): AppearanceContext => ({ task_id: task, ready: true, appearance: skill ? { skill_id: skill, revision, label: skill, states: {} } : null });
describe("stable mascot appearances", () => {
  it("waits for admission, holds the leading outfit across activity, and allows resource updates", () => {
    let latch = latchAppearance(BASE_LATCH, { task_id: "task-1", ready: false, appearance: null });
    expect(latch.decided).toBe(false);
    latch = latchAppearance(latch, context("doctor"));
    expect(latch.skill).toBe("doctor");
    expect(latchAppearance(latch, context("doctor", "task-1", "v2"))).toBe(latch);
    expect(latchAppearance(latch, context("coach", "task-2")).skill).toBe("coach");
    expect(latchAppearance(latch, BASE_CONTEXT)).toBe(BASE_LATCH);
  });
  it("falls back on ambiguity, removal or changing leaders without rapid costume switching", () => {
    for (const changed of [context(null), context("coach"), { ...context(null), ready: false }]) {
      let latch = latchAppearance(BASE_LATCH, context("doctor"));
      latch = latchAppearance(latch, changed);
      expect(latch.skill).toBeNull();
      expect(latchAppearance(latch, context("doctor")).skill).toBeNull();
    }
    const ambiguous = latchAppearance(BASE_LATCH, context(null));
    expect(latchAppearance(ambiguous, context("doctor")).skill).toBeNull();
  });
  it("persists per-user global and per-skill choices and tolerates damaged storage", () => {
    const values = new Map<string, string>();
    const storage = { getItem: (key: string) => values.get(key) || null, setItem: (key: string, value: string) => { values.set(key, value); } };
    saveAppearancePreferences(storage, "alex", { enabled: true, disabledSkills: ["doctor"] });
    expect(appearanceAllowed(readAppearancePreferences(storage, "alex"), "doctor")).toBe(false);
    expect(appearanceAllowed(readAppearancePreferences(storage, "alex"), "coach")).toBe(true);
    expect(appearanceAllowed(readAppearancePreferences(storage, "member"), "doctor")).toBe(true);
    saveAppearancePreferences(storage, "member", { enabled: false, disabledSkills: [] });
    expect(appearanceAllowed(readAppearancePreferences(storage, "member"), "coach")).toBe(false);
    expect(readAppearancePreferences({ getItem: () => "broken" }, "alex").enabled).toBe(true);
    saveAppearancePreferences(storage, "", { enabled: false, disabledSkills: [] });
    expect(values.size).toBe(2);
  });
});
