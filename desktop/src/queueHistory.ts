import type { QueueStatus } from "./pollState";

export const QUEUE_HISTORY_WINDOW_MS = 30 * 60 * 1000;

export type QueueSample = QueueStatus & { at: number };

export function queueWorkload(sample: Pick<QueueSample, "ready" | "processing">): number {
  return sample.ready + sample.processing;
}

export function appendQueueSample(
  current: QueueSample[],
  status: QueueStatus,
  at: number,
  windowMs = QUEUE_HISTORY_WINDOW_MS,
): QueueSample[] {
  const sample: QueueSample = {
    at,
    ready: Math.max(0, Math.floor(status.ready)),
    processing: Math.max(0, Math.floor(status.processing)),
  };
  const cutoff = at - windowMs;
  return [...current.filter((item) => item.at >= cutoff), sample];
}
