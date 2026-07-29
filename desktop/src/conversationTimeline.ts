import type { ChatMessage } from "./types";

export type ConversationTimelineItem =
  | { kind: "message"; key: string; message: ChatMessage }
  | { kind: "progress"; key: string; taskId: string };

export function buildConversationTimeline(
  messages: ChatMessage[],
  activeProgressTaskIds: string[],
  pendingProgressMessages: Record<string, ChatMessage>,
  morphedMessageTaskIds: Record<string, string>,
): ConversationTimelineItem[] {
  const pendingMessageIds = new Set(Object.values(pendingProgressMessages).map((message) => message.id));
  return [
    ...messages
      .filter((message) => !pendingMessageIds.has(message.id))
      .map((message): ConversationTimelineItem => {
        const taskId = morphedMessageTaskIds[message.id];
        return {
          kind: "message",
          key: taskId ? `task:${taskId}` : `message:${message.id}`,
          message,
        };
      }),
    ...activeProgressTaskIds.map((taskId): ConversationTimelineItem => ({
      kind: "progress",
      key: `task:${taskId}`,
      taskId,
    })),
  ];
}
