import type { Question } from "./types";
import type { A2uiSurface } from "./a2ui";

export type QueueStatus = { ready: number; processing: number };

export function reuseQuestions(current: Question[], next: Question[]): Question[] {
  if (current.length !== next.length) return next;
  const unchanged = current.every((question, index) => {
    const candidate = next[index];
    return JSON.stringify(question) === JSON.stringify(candidate);
  });
  return unchanged ? current : next;
}

export function reuseQueueStatus(current: QueueStatus, next: QueueStatus): QueueStatus {
  return current.ready === next.ready && current.processing === next.processing ? current : next;
}

export function withoutTaskProgressSurfaces(surfaces: Record<string, A2uiSurface>): Record<string, A2uiSurface> {
  return Object.fromEntries(Object.entries(surfaces).filter(([surfaceId]) => !surfaceId.startsWith("task-progress:")));
}

// Pending questions are the authoritative snapshot, including after reconnect
// or a dropped delete event. Never keep an actionable stale question surface.
export function reconcileQuestionSurfaces(surfaces: Record<string, A2uiSurface>, questions: Question[]): Record<string, A2uiSurface> {
  const pending = new Set(questions.map((question) => `question:${question.question_id}`));
  return Object.fromEntries(Object.entries(surfaces).filter(([id]) => !id.startsWith("question:") || pending.has(id)));
}
