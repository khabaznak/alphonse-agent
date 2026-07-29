import { describe, expect, it } from "vitest";
import { mergeFreshConversationHistory, orderConversationMessages } from "./conversationHistory";
import type { ChatMessage } from "./types";

describe("mergeFreshConversationHistory", () => {
  it("uses freshly loaded cross-channel history instead of a stale project cache", () => {
    const fresh: ChatMessage[] = [
      { id: "desktop-1", role: "user", content: "Earlier Desktop turn", source: "desktop" },
      { id: "telegram-1", role: "user", content: "New Telegram turn", source: "telegram-home" },
      { id: "telegram-2", role: "assistant", content: "New Telegram response", source: "telegram-home" },
    ];

    expect(mergeFreshConversationHistory(fresh, [])).toEqual(fresh);
  });

  it("preserves a message received while history is being reloaded without duplicating it", () => {
    const fresh: ChatMessage[] = [
      { id: "telegram-1", role: "user", content: "Telegram turn", source: "telegram-home" },
    ];
    const pending: ChatMessage[] = [
      { id: "desktop-local", role: "user", content: "Desktop turn sent during reload" },
      { id: "telegram-1", role: "user", content: "Telegram turn", source: "telegram-home" },
    ];

    expect(mergeFreshConversationHistory(fresh, pending)).toEqual([
      ...fresh,
      pending[0],
    ]);
  });

  it("orders user, steering, and assistant messages by authoritative sequence", () => {
    const messages: ChatMessage[] = [
      { id: "welcome", role: "assistant", content: "Welcome" },
      { id: "user-2", role: "user", content: "Steering", sequence: 12 },
      { id: "assistant-1", role: "assistant", content: "Done", sequence: 13 },
      { id: "user-1", role: "user", content: "Start", sequence: 11 },
    ];

    expect(orderConversationMessages(messages).map((message) => message.id)).toEqual([
      "welcome",
      "user-1",
      "user-2",
      "assistant-1",
    ]);
  });

  it("keeps repeated identical messages when their canonical ids differ", () => {
    const repeated: ChatMessage[] = [
      { id: "one", role: "user", content: "Continue", sequence: 1 },
      { id: "two", role: "user", content: "Continue", sequence: 2 },
    ];

    expect(mergeFreshConversationHistory([repeated[0]], [repeated[1]])).toEqual(repeated);
  });

  it("deduplicates the same live and historical message by canonical id", () => {
    const historical: ChatMessage = { id: "assistant-1", role: "assistant", content: "Done", sequence: 2 };
    const live: ChatMessage = { ...historical, source: "desktop" };

    expect(mergeFreshConversationHistory([historical], [live])).toEqual([historical]);
  });
});
