import { describe, expect, it } from "vitest";
import { conversationSendLabel, readConversationSendMode, writeConversationSendMode } from "./conversationSendMode";

describe("conversation send preference", () => {
  it("defaults to a follow-up and retains separate device choices for household members", () => {
    const values = new Map<string, string>();
    const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
    expect(readConversationSendMode(storage, "alex")).toBe("queue");
    writeConversationSendMode(storage, "alex", "steer");
    expect(readConversationSendMode(storage, "alex")).toBe("steer");
    expect(readConversationSendMode(storage, "sam")).toBe("queue");
    expect(conversationSendLabel(readConversationSendMode(storage, "alex"))).toBe("Send as update to this task");
    expect(conversationSendLabel(readConversationSendMode(storage, "sam"))).toBe("Send after this task");
  });
  it("uses a safe default for an invalid or absent preference", () => {
    expect(readConversationSendMode({ getItem: () => "unknown" }, "alex")).toBe("queue");
    expect(readConversationSendMode({ getItem: () => "steer" }, "")).toBe("queue");
  });
});
