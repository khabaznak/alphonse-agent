import type { ChatMessage } from "./types";
import { orderConversationMessages } from "./conversationHistory";
import type { CalendarEditReceipt } from "./CalendarEditReviews";

export type ConversationTimelineItem =
  | { kind: "message"; key: string; message: ChatMessage }
  | { kind: "progress"; key: string; taskId: string }
  | { kind: "calendar-receipt"; key: string; receipt: CalendarEditReceipt };

export function buildConversationTimeline(
  messages: ChatMessage[],
  activeProgressTaskIds: string[],
  pendingProgressMessages: Record<string, ChatMessage>,
  morphedMessageTaskIds: Record<string, string>,
  archivedProgressTaskIds: string[] = [],
  progressMetadata: Record<string, { created_at?: string; sequence?: number }> = {},
  calendarReceipts: CalendarEditReceipt[] = [],
): ConversationTimelineItem[] {
  const pendingMessageIds = new Set(Object.values(pendingProgressMessages).map((message) => message.id));
  const visibleMessages = messages.filter((message) => !pendingMessageIds.has(message.id));
  const anchored = new Set<string>();
  const items: ConversationTimelineItem[] = [];
  for (const message of visibleMessages) {
    const taskId = morphedMessageTaskIds[message.id];
    items.push({ kind: "message", key: taskId ? `task:${taskId}` : `message:${message.id}`, message });
    // Keep a task log beside its initiating request when their IDs match,
    // even after later messages or acknowledgements arrive from another task.
    if (message.role === "user" && activeProgressTaskIds.includes(message.id)) {
      items.push({ kind: "progress", key: `task:${message.id}`, taskId: message.id });
      anchored.add(message.id);
    }
  }
  // An unanchored log uses the initiating request's occurrence time. If that
  // metadata is unavailable, the same deterministic missing-time rule applies.
  const unanchored = activeProgressTaskIds.filter((id) => !anchored.has(id));
  const receipts = new Map(calendarReceipts.map(receipt => [`\u0000calendar-receipt:${receipt.id}`, receipt]));
  const ordered = orderConversationMessages([
    ...visibleMessages,
    ...unanchored.map((id): ChatMessage => ({ id: `\u0000progress:${id}`, role: "system", content: "", ...progressMetadata[id] })),
    ...Array.from(receipts, ([id, receipt]): ChatMessage => ({ id, role: "system", content: "", created_at: receipt.approved_at })),
  ]);
  const byMessage = new Map(items.filter((item) => item.kind === "message").map((item) => [item.kind === "message" ? item.message.id : "", item]));
  const progressByAnchor = new Map(items.filter((item) => item.kind === "progress").map((item) => [item.kind === "progress" ? item.taskId : "", item]));
  return ordered.flatMap<ConversationTimelineItem>((message) => {
    if (message.id.startsWith("\u0000progress:")) {
      const taskId = message.id.slice("\u0000progress:".length);
      return [{ kind: "progress" as const, key: `task:${taskId}`, taskId }];
    }
    const receipt = receipts.get(message.id);
    if (receipt) return [{ kind: "calendar-receipt" as const, key: `calendar-receipt:${receipt.id}`, receipt }];
    const item = byMessage.get(message.id);
    if (!item) return [];
    const progress = progressByAnchor.get(message.id);
    return progress ? [item, progress] : [item];
  });
}
