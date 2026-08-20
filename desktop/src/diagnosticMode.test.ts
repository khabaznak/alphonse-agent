import { describe, expect, it } from "vitest";
import { desktopDiagnosticBehavior, parseDesktopDiagnosticMode } from "./diagnosticMode";

describe("desktop diagnostic modes", () => {
  it("parses supported modes and normalizes surrounding text", () => {
    expect(parseDesktopDiagnosticMode(" static ")).toBe("static");
    expect(parseDesktopDiagnosticMode("PING-ONLY")).toBe("ping-only");
    expect(parseDesktopDiagnosticMode("poll-no-commit")).toBe("poll-no-commit");
    expect(parseDesktopDiagnosticMode("render-only")).toBe("render-only");
    expect(parseDesktopDiagnosticMode("history-render-memo")).toBe("history-render-memo");
    expect(parseDesktopDiagnosticMode("history-render-timeline-memo")).toBe("history-render-timeline-memo");
  });

  it("falls back to normal for missing and unknown values", () => {
    expect(parseDesktopDiagnosticMode(undefined)).toBe("normal");
    expect(parseDesktopDiagnosticMode("something-else")).toBe("normal");
  });

  it.each([
    ["static", false, "none", false, false, false, false, false],
    ["ping-only", true, "ping", false, false, false, false, false],
    ["poll-no-commit", true, "poll", false, false, false, false, false],
    ["render-only", false, "render", false, false, false, false, false],
    ["history-static", true, "none", false, true, false, false, false],
    ["history-render", true, "render", false, true, false, false, false],
    ["history-render-plain", true, "render", false, true, true, false, false],
    ["history-render-memo", true, "render", false, true, false, true, false],
    ["history-render-timeline-memo", true, "render", false, true, false, true, true],
    ["normal", true, "poll", true, false, false, true, true],
  ] as const)("configures %s without enabling unintended work", (mode, startsDaemon, cycle, commitsPollResponses, loadsHistory, plainTextMessages, memoizesMessages, memoizesTimeline) => {
    expect(desktopDiagnosticBehavior(mode)).toEqual({ startsDaemon, cycle, commitsPollResponses, loadsHistory, plainTextMessages, memoizesMessages, memoizesTimeline });
  });
});
