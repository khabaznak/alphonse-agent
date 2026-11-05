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

it("keeps an older task log with its request when a receipt and acknowledgement arrive", () => {
  const items = buildConversationTimeline([
    { id: "repair", role: "user", content: "Repair the artifact", sequence: 1 },
    { id: "failed", role: "assistant", content: "Could not repair it", sequence: 2 },
    { id: "receipt", role: "user", content: "Dinner receipt", sequence: 3 },
    { id: "ack", role: "assistant", content: "I will read the receipt", sequence: 4 },
  ], ["repair", "receipt"], {}, {});
  expect(items.map((item) => item.key)).toEqual([
    "message:repair", "task:repair", "message:failed", "message:receipt", "task:receipt", "message:ack",
  ]);
});

it("places an archived task from a previous day before today's receipt conversation", () => {
  const items = buildConversationTimeline([
    { id: "receipt", role: "user", content: "Dinner receipt" },
    { id: "ack", role: "assistant", content: "I will read the receipt" },
  ], ["yesterday-repair", "receipt"], {}, {}, ["yesterday-repair"]);
  expect(items.map((item) => item.key)).toEqual([
    "task:yesterday-repair", "message:receipt", "task:receipt", "message:ack",
  ]);
});
