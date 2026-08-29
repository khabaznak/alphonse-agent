import type { ActivityEvent } from "./types";

export const HOME_PROJECT_KEY = "__home__";
export type AvatarState = "idle" | "planning" | "doing" | "checking" | "acting" | "error" | "disconnected";
const AVATAR_STATE_LABELS: Record<AvatarState, string> = {
  idle: "Idle",
  planning: "Planning",
  doing: "Doing",
  checking: "Checking",
  acting: "Acting",
  error: "Error",
  disconnected: "Disconnected",
};

export function projectKey(projectId?: string): string {
  return projectId && projectId.trim() ? projectId.trim() : HOME_PROJECT_KEY;
}

export function capdActivityLabel(event?: Pick<ActivityEvent, "phase" | "label"> | null, fallback = "idle"): string {
  const raw = `${event?.label || ""} ${event?.phase || ""} ${fallback || ""}`.toLowerCase();
  if (raw.includes("check")) return "checking";
  if (raw.includes("plan") || raw.includes("deliberat") || raw.includes("decid")) return "planning";
  if (raw.includes("act")) return "acting";
  if (raw.includes("do") || raw.includes("execute") || raw.includes("tool")) return "doing";
  return fallback.toLowerCase().includes("work") ? "doing" : "idle";
}

export function avatarState(connected: boolean, error: string, activity: string): AvatarState {
  if (!connected) return "disconnected";
  if (error.trim()) return "error";
  if (activity === "planning" || activity === "doing" || activity === "checking" || activity === "acting") return activity;
  return "idle";
}

export function avatarStateLabel(state: AvatarState): string {
  return AVATAR_STATE_LABELS[state];
}
