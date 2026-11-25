import { expect, it } from "vitest";
import { readConversationFailures, saveConversationFailures } from "./conversationFailures";

function disposableStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() { return values.size; },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    removeItem: (key) => { values.delete(key); },
    setItem: (key, value) => { values.set(key, value); },
  };
}

it("persists observed failures and dismissal per household member without storing message content", () => {
  const storage = disposableStorage();
  const record = { messageId: "a".repeat(32), failureCode: "source_unavailable", dismissed: true };
  expect(saveConversationFailures(storage, "alice", [record])).toBe(true);
  expect(readConversationFailures(storage, "alice")).toEqual([record]);
  expect(readConversationFailures(storage, "bob")).toEqual([]);
  expect(storage.getItem("alphonse.desktop.conversationFailures.alice")).not.toContain("boiler");
});

it("ignores invalid local data and handles unavailable storage", () => {
  const storage = disposableStorage();
  storage.setItem("alphonse.desktop.conversationFailures.alice", JSON.stringify([{ messageId: "bad", failureCode: "source_unavailable", dismissed: false }]));
  expect(readConversationFailures(storage, "alice")).toEqual([]);
  const broken = { getItem: () => { throw new Error(); }, setItem: () => { throw new Error(); } } as unknown as Storage;
  expect(readConversationFailures(broken, "alice")).toEqual([]);
  expect(saveConversationFailures(broken, "alice", [])).toBe(false);
});
