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
  const sequenced = messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) => typeof message.sequence === "number" && message.sequence > 0)
    .sort((left, right) => (left.message.sequence || 0) - (right.message.sequence || 0) || left.index - right.index)
    .map(({ message }) => message);
  let sequenceIndex = 0;
  return messages.map((message) => {
    if (typeof message.sequence !== "number" || message.sequence <= 0) return message;
    return sequenced[sequenceIndex++];
  });
}
