import type { ChatMessage } from "./types";

export function mergeFreshConversationHistory(
  history: ChatMessage[],
  messagesReceivedDuringReload: ChatMessage[],
): ChatMessage[] {
  const merged = [...history];
  const knownIds = new Set(merged.map((message) => message.id));
  for (const message of messagesReceivedDuringReload) {
    if (knownIds.has(message.id)) continue;
    knownIds.add(message.id);
    merged.push(message);
  }
  return orderConversationMessages(merged);
}

export function orderConversationMessages(messages: ChatMessage[]): ChatMessage[] {
  return [...messages].sort(conversationOrder);
}

// Invalid or absent timestamps sort after dated events. Sequence and ID make
// ties deterministic across polling and reloads, regardless of input order.
function conversationOrder(left: ChatMessage, right: ChatMessage): number {
  const leftTime = Date.parse(left.created_at || "");
  const rightTime = Date.parse(right.created_at || "");
  const a = Number.isFinite(leftTime) ? leftTime : Infinity;
  const b = Number.isFinite(rightTime) ? rightTime : Infinity;
  if (a !== b) return a < b ? -1 : 1;
  const leftSequence = left.sequence && left.sequence > 0 ? left.sequence : Infinity;
  const rightSequence = right.sequence && right.sequence > 0 ? right.sequence : Infinity;
  if (leftSequence !== rightSequence) return leftSequence < rightSequence ? -1 : 1;
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
}
