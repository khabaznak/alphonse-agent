import { describe, expect, it } from "vitest";
import { reuseQuestions, reuseQueueStatus, withoutTaskProgressSurfaces } from "./pollState";

describe("idle desktop poll state", () => {
  it("preserves question identity when contents are unchanged", () => {
    const current = [{ question_id: "q1", task_id: "task-1", project_id: "p1", message: "Continue?", kind: "yes_no" as const, choices: [] }];
    expect(reuseQuestions(current, current.map((question) => ({ ...question })))).toBe(current);
    expect(reuseQuestions(current, [{ ...current[0], message: "Ready?" }])).not.toBe(current);
  });

  it("preserves queue identity when counts are unchanged", () => {
    const current = { ready: 0, processing: 0 };
    expect(reuseQueueStatus(current, { ready: 0, processing: 0 })).toBe(current);
    expect(reuseQueueStatus(current, { ready: 1, processing: 0 })).not.toBe(current);
  });

  it("clears stale task progress while retaining other surfaces after a daemon restart", () => {
    const question = { surfaceId: "question:q1", catalogId: "alphonse.desktop.catalog.v2", components: {}, dataModel: {} };
    const progress = { surfaceId: "task-progress:t1", catalogId: "alphonse.desktop.catalog.v2", components: {}, dataModel: {} };
    expect(withoutTaskProgressSurfaces({ [question.surfaceId]: question, [progress.surfaceId]: progress })).toEqual({
      [question.surfaceId]: question,
    });
  });
});

import { reconcileQuestionSurfaces } from "./pollState";
it("drops stale question controls from the authoritative poll even if delete events were lost", () => {
  const surface = { surfaceId: "question:old", catalogId: "alphonse.desktop.catalog.v2", components: {}, dataModel: {} };
  expect(reconcileQuestionSurfaces({"question:old":surface,"scheduled-task:one":{...surface,surfaceId:"scheduled-task:one"}},[])).toEqual({"scheduled-task:one":{...surface,surfaceId:"scheduled-task:one"}});
});
it("updates expiry and context revocations without resetting unchanged question records", () => {
  const q = { question_id:"q",task_id:"t",project_id:"",message:"Fixture",kind:"open_text" as const,choices:[],context:{text:"Context"} };
  const current=[q];
  expect(reuseQuestions(current,[{...q,context:{text:"Updated"}}])).not.toBe(current);
  expect(reuseQuestions(current,[{...q,expires_at:"2026-10-30T00:00:00Z"}])).not.toBe(current);
});
