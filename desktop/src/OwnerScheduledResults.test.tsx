import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { awaitingPresentation, mergeScheduledResults, OwnerScheduledResults, scheduledResultMessage } from "./OwnerScheduledResults";

describe("scheduled result presentation", () => {
  it("maps a result to the same identity used by conversation history", () => {
    expect(scheduledResultMessage({ id: "abc", message: "Four cans", kind: "answer",
      created_at: "2026-01-01T12:00:00Z", sequence: 4 })).toEqual({
      id: "scheduled:abc", role: "assistant", source: "scheduled", content: "Four cans",
      created_at: "2026-01-01T12:00:00Z", sequence: 4,
    });
  });

  it("does not acknowledge during server rendering or while inactive", () => {
    const request = vi.fn();
    const onPresented = vi.fn();
    const html = renderToStaticMarkup(<OwnerScheduledResults request={request} active={false} onPresented={onPresented} />);
    expect(html).toBe("");
    expect(request).not.toHaveBeenCalled();
    expect(onPresented).not.toHaveBeenCalled();
  });

  it("makes every pending result available without waiting for dismissal", () => {
    const first = { id: "a", message: "One", kind: "answer" as const, created_at: "2026-01-01T12:00:00Z", sequence: 1 };
    const second = { ...first, id: "b", message: "Two", sequence: 2 };
    const results = mergeScheduledResults([], [first, second], new Set());
    expect(awaitingPresentation(results, true, new Set()).map(item => item.id)).toEqual(["a", "b"]);
    expect(awaitingPresentation(results, false, new Set())).toEqual([]);
    expect(awaitingPresentation(results, true, new Set(["a"])).map(item => item.id)).toEqual(["b"]);
    expect(mergeScheduledResults(results, [first, second], new Set())).toHaveLength(2);
    expect(mergeScheduledResults([], [first, second], new Set(["a"])).map(item => item.id)).toEqual(["b"]);
  });
});
