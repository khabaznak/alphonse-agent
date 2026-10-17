import type { ActivityEvent } from "./types";

export const PDCA_HISTORY_WINDOW_MS = 30 * 60 * 1000;

export type PdcaPhase = "plan" | "do" | "check" | "act";
export type PdcaSample = { at: number; phase: PdcaPhase | null };

export function pdcaPhase(value: string): PdcaPhase | null {
  const normalized = value.trim().toLowerCase();
  return normalized === "plan" || normalized === "do" || normalized === "check" || normalized === "act"
    ? normalized
    : null;
}

function eventTime(event: Pick<ActivityEvent, "occurred_at">, fallback: number): number {
  const parsed = Date.parse(event.occurred_at || "");
  return Number.isFinite(parsed) ? Math.min(parsed, fallback) : fallback;
}

function appendTransition(samples: PdcaSample[], sample: PdcaSample): PdcaSample[] {
  const previous = samples.at(-1);
  if (previous?.phase === sample.phase) return samples;
  return [...samples, { ...sample, at: Math.max(previous?.at || 0, sample.at) }];
}

export function appendPdcaActivity(
  current: PdcaSample[],
  events: Array<Pick<ActivityEvent, "phase" | "occurred_at">>,
  activityState: string,
  at: number,
  windowMs = PDCA_HISTORY_WINDOW_MS,
): PdcaSample[] {
  let next = [...current];
  for (const event of events) {
    const phase = pdcaPhase(event.phase);
    if (phase) next = appendTransition(next, { at: eventTime(event, at), phase });
  }

  const working = activityState.trim().toLowerCase() === "working";
  if (!working) next = appendTransition(next, { at, phase: null });

  const cutoff = at - windowMs;
  const firstInside = next.findIndex((sample) => sample.at >= cutoff);
  if (firstInside <= 0) return firstInside === 0 ? next : next.slice(-1);
  return next.slice(firstInside - 1);
}

export function phaseAt(samples: PdcaSample[], at: number): PdcaPhase | null {
  let phase: PdcaPhase | null = null;
  for (const sample of samples) {
    if (sample.at > at) break;
    phase = sample.phase;
  }
  return phase;
}
