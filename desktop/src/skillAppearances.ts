import type { AvatarState } from "./layoutState";

export const MASCOT_STATES: AvatarState[] = ["idle", "planning", "doing", "checking", "acting", "error", "disconnected"];
export type SkillAppearance = {
  skill_id: string; label: string; revision: string; preset?: string;
  states: Partial<Record<AvatarState, { resource: string; frame?: [number, number, number, number] }>>;
  resources?: Record<string, string>;
};
export type AppearanceContext = { task_id: string; ready: boolean; appearance: SkillAppearance | null };
export const BASE_CONTEXT: AppearanceContext = { task_id: "", ready: false, appearance: null };
export type AppearanceLatch = { task: string; decided: boolean; skill: string | null; invalidated: boolean };
export const BASE_LATCH: AppearanceLatch = { task: "", decided: false, skill: null, invalidated: false };

// One outfit per task. If a later selection conflicts, stay in the base outfit
// for the remainder of that task. Resource updates for the same skill are fine.
export function latchAppearance(previous: AppearanceLatch, context: AppearanceContext): AppearanceLatch {
  if (!context.task_id) return BASE_LATCH;
  const current = previous.task === context.task_id ? previous : { ...BASE_LATCH, task: context.task_id };
  if (!context.ready) return { ...current, skill: null, invalidated: current.decided };
  const skill = context.appearance?.skill_id ?? null;
  if (!current.decided) return { ...current, decided: true, skill };
  if (current.invalidated || current.skill !== skill) return { ...current, skill: null, invalidated: true };
  return current;
}

export type AppearancePreferences = { enabled: boolean; disabledSkills: string[] };
export const DEFAULT_APPEARANCE_PREFERENCES: AppearancePreferences = { enabled: true, disabledSkills: [] };
export const appearancePreferencesKey = (user: string) => `alphonse.desktop.skillAppearances.${encodeURIComponent(user)}`;
export function readAppearancePreferences(storage: Pick<Storage, "getItem">, user: string): AppearancePreferences {
  try {
    const value = JSON.parse(storage.getItem(appearancePreferencesKey(user)) || "null");
    if (!value || typeof value.enabled !== "boolean" || !Array.isArray(value.disabledSkills)) return DEFAULT_APPEARANCE_PREFERENCES;
    return { enabled: value.enabled, disabledSkills: value.disabledSkills.filter((item: unknown) => typeof item === "string") };
  } catch { return DEFAULT_APPEARANCE_PREFERENCES; }
}
export function saveAppearancePreferences(storage: Pick<Storage, "setItem">, user: string, value: AppearancePreferences): void {
  if (user) storage.setItem(appearancePreferencesKey(user), JSON.stringify(value));
}
export function appearanceAllowed(preferences: AppearancePreferences, skill: string): boolean {
  return preferences.enabled && !preferences.disabledSkills.includes(skill);
}
