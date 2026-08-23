import type { Question } from "./types";

export type ProjectAttention = Record<string, { unread_messages: number; pending_questions: number; total: number }>;
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

export function reuseProjectAttention(current: ProjectAttention, next: ProjectAttention): ProjectAttention {
  const currentKeys = Object.keys(current);
  const nextKeys = Object.keys(next);
  if (currentKeys.length !== nextKeys.length) return next;
  const unchanged = currentKeys.every((projectId) => {
    const left = current[projectId];
    const right = next[projectId];
    return Boolean(right)
      && left.unread_messages === right.unread_messages
      && left.pending_questions === right.pending_questions
      && left.total === right.total;
  });
  return unchanged ? current : next;
}

export function reuseQueueStatus(current: QueueStatus, next: QueueStatus): QueueStatus {
  return current.ready === next.ready && current.processing === next.processing ? current : next;
}
