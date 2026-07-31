import type { A2uiSurface } from "./a2ui";

export const DISMISSED_SCHEDULED_SURFACES_KEY = "alphonse.desktop.dismissedScheduledTaskCards";
export const MAX_DISMISSED_SCHEDULED_SURFACES = 200;

export function readDismissedScheduledSurfaces(storage: Pick<Storage, "getItem">): Set<string> {
  try {
    const parsed = JSON.parse(storage.getItem(DISMISSED_SCHEDULED_SURFACES_KEY) || "[]");
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter(isScheduledSurfaceId).slice(-MAX_DISMISSED_SCHEDULED_SURFACES));
  } catch {
    return new Set();
  }
}

export function rememberDismissedScheduledSurface(storage: Pick<Storage, "setItem">, current: Set<string>, surfaceId: string): Set<string> {
  if (!isScheduledSurfaceId(surfaceId)) return current;
  const ids = [...current].filter((value) => value !== surfaceId);
  ids.push(surfaceId);
  const bounded = ids.slice(-MAX_DISMISSED_SCHEDULED_SURFACES);
  try { storage.setItem(DISMISSED_SCHEDULED_SURFACES_KEY, JSON.stringify(bounded)); } catch { /* UI dismissal still succeeds. */ }
  return new Set(bounded);
}

export function withoutDismissedSurfaces(surfaces: Record<string, A2uiSurface>, dismissed: Set<string>): Record<string, A2uiSurface> {
  if (!dismissed.size) return surfaces;
  const next = { ...surfaces };
  dismissed.forEach((surfaceId) => { delete next[surfaceId]; });
  return next;
}

export function withoutSurface(surfaces: Record<string, A2uiSurface>, surfaceId: string): Record<string, A2uiSurface> {
  if (!(surfaceId in surfaces)) return surfaces;
  const next = { ...surfaces };
  delete next[surfaceId];
  return next;
}

function isScheduledSurfaceId(value: unknown): value is string {
  return typeof value === "string" && value.startsWith("scheduled-task:") && value.length > "scheduled-task:".length;
}
