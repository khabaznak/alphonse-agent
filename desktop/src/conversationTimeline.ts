import type { ChatMessage } from "./types";

export type ConversationTimelineItem =
  | { kind: "message"; key: string; message: ChatMessage }
  | { kind: "progress"; key: string; taskId: string };

export function buildConversationTimeline(
  messages: ChatMessage[],
  activeProgressTaskIds: string[],
  pendingProgressMessages: Record<string, ChatMessage>,
  morphedMessageTaskIds: Record<string, string>,
  archivedProgressTaskIds: string[] = [],
): ConversationTimelineItem[] {
  const pendingMessageIds = new Set(Object.values(pendingProgressMessages).map((message) => message.id));
  const visibleMessages = messages.filter((message) => !pendingMessageIds.has(message.id));
  const anchored = new Set<string>();
  const items: ConversationTimelineItem[] = [];
  for (const message of visibleMessages) {
    const taskId = morphedMessageTaskIds[message.id];
    items.push({ kind: "message", key: taskId ? `task:${taskId}` : `message:${message.id}`, message });
    // V3 tasks use the inbound message ID. Keep each log beside its own request,
    // even after later messages or acknowledgements arrive from another task.
    if (message.role === "user" && activeProgressTaskIds.includes(message.id)) {
      items.push({ kind: "progress", key: `task:${message.id}`, taskId: message.id });
      anchored.add(message.id);
    }
  }
  const earlier: ConversationTimelineItem[] = [];
  for (const taskId of activeProgressTaskIds) {
    if (anchored.has(taskId)) continue;
    const item: ConversationTimelineItem = { kind: "progress", key: `task:${taskId}`, taskId };
    // An archived task may have started on a previous conversation day. Its
    // request is outside this page; do not attach its log to today's newest task.
    if (archivedProgressTaskIds.includes(taskId)) earlier.push(item); else items.push(item);
  }
  return [...earlier, ...items];
}
