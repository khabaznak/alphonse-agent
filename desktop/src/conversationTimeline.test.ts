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
    { id: "receipt", role: "user", content: "Dinner receipt", created_at: "2026-01-02T10:00:00Z" },
    { id: "ack", role: "assistant", content: "I will read the receipt", created_at: "2026-01-02T10:01:00Z" },
  ], ["yesterday-repair", "receipt"], {}, {}, ["yesterday-repair"], { "yesterday-repair": { created_at: "2026-01-01T00:00:00Z" } });
  expect(items.map((item) => item.key)).toEqual([
    "task:yesterday-repair", "message:receipt", "task:receipt", "message:ack",
  ]);
});

it("places unanchored active and archived logs by request time", () => {
  const messages: ChatMessage[] = [
    { id: "one", role: "user", content: "One", created_at: "2026-01-02T10:00:00Z" },
    { id: "two", role: "assistant", content: "Two", created_at: "2026-01-02T12:00:00Z" },
  ];
  const metadata = { earlier: { created_at: "2026-01-01T23:00:00Z" }, between: { created_at: "2026-01-02T11:00:00Z" } };
  expect(buildConversationTimeline(messages, ["between", "earlier"], {}, {}, ["earlier"], metadata).map((item) => item.key)).toEqual([
    "task:earlier", "message:one", "task:between", "message:two",
  ]);
});
