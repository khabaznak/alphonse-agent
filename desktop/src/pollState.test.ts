import { describe, expect, it } from "vitest";
import { reuseProjectAttention, reuseQuestions, reuseQueueStatus } from "./pollState";

describe("idle desktop poll state", () => {
  it("preserves question identity when contents are unchanged", () => {
    const current = [{ question_id: "q1", project_id: "p1", message: "Continue?", kind: "yes_no" as const, choices: [] }];
    expect(reuseQuestions(current, current.map((question) => ({ ...question })))).toBe(current);
    expect(reuseQuestions(current, [{ ...current[0], message: "Ready?" }])).not.toBe(current);
  });

  it("preserves project-attention identity when counts are unchanged", () => {
    const current = { p1: { unread_messages: 1, pending_questions: 2, total: 3 } };
    expect(reuseProjectAttention(current, { p1: { ...current.p1 } })).toBe(current);
    expect(reuseProjectAttention(current, { p1: { ...current.p1, total: 4 } })).not.toBe(current);
  });

  it("preserves queue identity when counts are unchanged", () => {
    const current = { ready: 0, processing: 0 };
    expect(reuseQueueStatus(current, { ready: 0, processing: 0 })).toBe(current);
    expect(reuseQueueStatus(current, { ready: 1, processing: 0 })).not.toBe(current);
  });
});
