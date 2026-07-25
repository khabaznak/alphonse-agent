import { describe, expect, it } from "vitest";
import { mergeFreshConversationHistory } from "./conversationHistory";
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
});
