import { describe, expect, it } from "vitest";
import { appendPdcaActivity, pdcaPhase, phaseAt } from "./pdcaHistory";

describe("PDCA activity history", () => {
  it("accepts only the four graph phases", () => {
    expect(pdcaPhase("plan")).toBe("plan");
    expect(pdcaPhase("CHECK")).toBe("check");
    expect(pdcaPhase("tool")).toBeNull();
  });

  it("records phase transitions and an idle boundary", () => {
    const working = appendPdcaActivity([], [
      { phase: "plan", occurred_at: "1970-01-01T00:00:01.000Z" },
      { phase: "do", occurred_at: "1970-01-01T00:00:02.000Z" },
    ], "working", 3_000, 10_000);
    expect(working).toEqual([
      { at: 1_000, phase: "plan" },
      { at: 2_000, phase: "do" },
    ]);

    const idle = appendPdcaActivity(working, [], "idle", 4_000, 10_000);
    expect(idle.at(-1)).toEqual({ at: 4_000, phase: null });
    expect(phaseAt(idle, 2_500)).toBe("do");
    expect(phaseAt(idle, 4_000)).toBeNull();
  });

  it("keeps the last boundary before the moving window", () => {
    const history = [
      { at: 1_000, phase: "plan" as const },
      { at: 2_000, phase: "do" as const },
      { at: 3_000, phase: "check" as const },
    ];
    expect(appendPdcaActivity(history, [], "working", 4_000, 1_500)).toEqual([
      { at: 2_000, phase: "do" },
      { at: 3_000, phase: "check" },
    ]);
  });
});
