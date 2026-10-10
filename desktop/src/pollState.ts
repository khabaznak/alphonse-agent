import type { Question } from "./types";
import type { A2uiSurface } from "./a2ui";

export type QueueStatus = { ready: number; processing: number };

export function reuseQuestions(current: Question[], next: Question[]): Question[] {
  if (current.length !== next.length) return next;
  const unchanged = current.every((question, index) => {
    const candidate = next[index];
    return question.question_id === candidate.question_id
      && question.task_id === candidate.task_id
      && question.project_id === candidate.project_id
      && question.created_at === candidate.created_at
      && question.conversation_sequence === candidate.conversation_sequence
      && question.message === candidate.message
      && question.kind === candidate.kind
      && question.choices.length === candidate.choices.length
      && question.choices.every((choice, choiceIndex) => choice.id === candidate.choices[choiceIndex].id && choice.label === candidate.choices[choiceIndex].label);
  });
  return unchanged ? current : next;
}

export function reuseQueueStatus(current: QueueStatus, next: QueueStatus): QueueStatus {
  return current.ready === next.ready && current.processing === next.processing ? current : next;
}

export function withoutTaskProgressSurfaces(surfaces: Record<string, A2uiSurface>): Record<string, A2uiSurface> {
  return Object.fromEntries(Object.entries(surfaces).filter(([surfaceId]) => !surfaceId.startsWith("task-progress:")));
}
