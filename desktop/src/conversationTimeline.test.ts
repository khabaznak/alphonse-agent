import { describe, expect, it } from "vitest";
import { buildConversationTimeline } from "./conversationTimeline";
import type { ChatMessage } from "./types";

describe("buildConversationTimeline", () => {
  const user: ChatMessage = { id: "user-1", role: "user", content: "Start", sequence: 1 };
  const final: ChatMessage = { id: "assistant-1", role: "assistant", content: "Done", sequence: 2 };

  it("shows the CAPD entry while its final response is pending", () => {
    const items = buildConversationTimeline(
      [user, final],
      ["task-1"],
      { "task-1": final },
      {},
    );

    expect(items.map((item) => [item.kind, item.key])).toEqual([
      ["message", "message:user-1"],
      ["progress", "task:task-1"],
    ]);
  });

  it("graduates a completed CAPD response into the normal timeline exactly once", () => {
    const items = buildConversationTimeline(
      [user, final],
      [],
      {},
      { "assistant-1": "task-1" },
    );

    expect(items.map((item) => [item.kind, item.key])).toEqual([
      ["message", "message:user-1"],
      ["message", "task:task-1"],
    ]);
    expect(items.filter((item) => item.kind === "message" && item.message.id === final.id)).toHaveLength(1);
  });
});
