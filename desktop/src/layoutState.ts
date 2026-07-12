import type { ActivityEvent } from "./types";

export const HOME_PROJECT_KEY = "__home__";

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

export function agentStateLabel(connected: boolean, hasError: boolean, activeWorkCount: number): "Idle" | "Working" | "Error" | "Disconnected" {
  if (!connected) return "Disconnected";
  if (hasError) return "Error";
  return activeWorkCount > 0 ? "Working" : "Idle";
}
