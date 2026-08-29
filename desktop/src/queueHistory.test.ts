import { describe, expect, it } from "vitest";
import { appendQueueSample, queueWorkload } from "./queueHistory";

describe("queue workload history", () => {
  it("counts waiting and processing tasks as workload", () => {
    expect(queueWorkload({ ready: 2, processing: 3 })).toBe(5);
  });

  it("keeps only samples inside the moving window", () => {
    const history = [
      { at: 1_000, ready: 1, processing: 0 },
      { at: 2_000, ready: 1, processing: 1 },
    ];
    expect(appendQueueSample(history, { ready: 2, processing: 1 }, 3_000, 1_500)).toEqual([
      { at: 2_000, ready: 1, processing: 1 },
      { at: 3_000, ready: 2, processing: 1 },
    ]);
  });

  it("normalizes invalid negative or fractional counts", () => {
    expect(appendQueueSample([], { ready: -1, processing: 2.9 }, 1_000)).toEqual([
      { at: 1_000, ready: 0, processing: 2 },
    ]);
  });
});
