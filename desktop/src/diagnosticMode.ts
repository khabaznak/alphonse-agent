export const DESKTOP_DIAGNOSTIC_MODES = [
  "normal",
  "static",
  "ping-only",
  "poll-no-commit",
  "render-only",
  "history-static",
  "history-render",
  "history-render-plain",
  "history-render-memo",
  "history-render-timeline-memo",
] as const;

export type DesktopDiagnosticMode = (typeof DESKTOP_DIAGNOSTIC_MODES)[number];
export type DiagnosticCycle = "none" | "ping" | "poll" | "render";

export type DesktopDiagnosticBehavior = {
  startsDaemon: boolean;
  cycle: DiagnosticCycle;
  commitsPollResponses: boolean;
  loadsHistory: boolean;
  plainTextMessages: boolean;
  memoizesMessages: boolean;
  memoizesTimeline: boolean;
};

export function parseDesktopDiagnosticMode(value: unknown): DesktopDiagnosticMode {
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
  return DESKTOP_DIAGNOSTIC_MODES.includes(normalized as DesktopDiagnosticMode)
    ? normalized as DesktopDiagnosticMode
    : "normal";
}

export function desktopDiagnosticBehavior(mode: DesktopDiagnosticMode): DesktopDiagnosticBehavior {
  switch (mode) {
    case "static":
      return behavior(false, "none");
    case "ping-only":
      return behavior(true, "ping");
    case "poll-no-commit":
      return behavior(true, "poll");
    case "render-only":
      return behavior(false, "render");
    case "history-static":
      return behavior(true, "none", { loadsHistory: true });
    case "history-render":
      return behavior(true, "render", { loadsHistory: true });
    case "history-render-plain":
      return behavior(true, "render", { loadsHistory: true, plainTextMessages: true });
    case "history-render-memo":
      return behavior(true, "render", { loadsHistory: true, memoizesMessages: true });
    case "history-render-timeline-memo":
      return behavior(true, "render", { loadsHistory: true, memoizesMessages: true, memoizesTimeline: true });
    default:
      return behavior(true, "poll", { commitsPollResponses: true, memoizesMessages: true, memoizesTimeline: true });
  }
}

function behavior(
  startsDaemon: boolean,
  cycle: DiagnosticCycle,
  overrides: Partial<DesktopDiagnosticBehavior> = {},
): DesktopDiagnosticBehavior {
  return {
    startsDaemon,
    cycle,
    commitsPollResponses: false,
    loadsHistory: false,
    plainTextMessages: false,
    memoizesMessages: false,
    memoizesTimeline: false,
    ...overrides,
  };
}
