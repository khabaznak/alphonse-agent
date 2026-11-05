import { applyA2uiEvent, type A2uiSurface, type AgUiEvent } from "./a2ui";

const PREFIX = "task-progress:";

export const terminalProgressStates = new Set(["completed", "done", "success", "failed", "cancelled", "canceled", "abandoned", "interrupted"]);
export function progressStatus(surface: A2uiSurface): string {
  return String(surface.dataModel.status || "").toLowerCase();
}

// Each task owns its activity; closures preserve the last snapshot as a work log.
export function projectTaskProgress(
  current: Record<string, A2uiSurface>,
  archived: Record<string, A2uiSurface>,
  events: Array<{ event: AgUiEvent }>,
  stopped: boolean,
) {
  let surfaces = current;
  const logs = { ...archived };

  const remember = (id: string) => {
    if (surfaces[id]) logs[id.slice(PREFIX.length)] = surfaces[id];
  };
  for (const { event } of events) {
    const next = applyA2uiEvent(surfaces, event);
    for (const id of Object.keys(surfaces)) {
      if (id.startsWith(PREFIX) && !next[id]) {
        remember(id);
      }
    }
    for (const id of Object.keys(next)) {
      if (!id.startsWith(PREFIX) || next[id] === surfaces[id]) continue;
      if (next[id].dataModel.status && !terminalProgressStates.has(progressStatus(next[id]))) delete logs[id.slice(PREFIX.length)];
    }
    surfaces = next;
  }
  // Also recover stale cards after failures that emitted no closing envelope.
  for (const id of Object.keys(surfaces)) {
    if (!id.startsWith(PREFIX)) continue;
    if (terminalProgressStates.has(progressStatus(surfaces[id]))) remember(id);
    else if (stopped && !logs[id.slice(PREFIX.length)]) logs[id.slice(PREFIX.length)] = {
      ...surfaces[id], dataModel: { ...surfaces[id].dataModel, status: "interrupted" },
    };
  }
  return { surfaces, archived: logs };
}
