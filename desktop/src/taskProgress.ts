import { applyA2uiEvent, type A2uiSurface, type AgUiEvent } from "./a2ui";

const PREFIX = "task-progress:";

// Keep the last snapshot even when deleteSurface arrives without a delivery.
export function projectTaskProgress(
  current: Record<string, A2uiSurface>,
  archived: Record<string, A2uiSurface>,
  events: Array<{ event: AgUiEvent }>,
  stopped: boolean,
) {
  let surfaces = current;
  const logs = { ...archived };
  let activeId = Object.keys(current).filter((id) => id.startsWith(PREFIX) && !logs[id.slice(PREFIX.length)]).at(-1);
  const remember = (id: string) => {
    if (surfaces[id]) logs[id.slice(PREFIX.length)] = surfaces[id];
  };
  for (const { event } of events) {
    const next = applyA2uiEvent(surfaces, event);
    for (const id of Object.keys(surfaces)) {
      if (id.startsWith(PREFIX) && !next[id]) {
        remember(id);
        if (activeId === id) activeId = undefined;
      }
    }
    for (const id of Object.keys(next)) {
      if (!id.startsWith(PREFIX) || next[id] === surfaces[id]) continue;
      if (activeId && activeId !== id) remember(activeId);
      activeId = id;
      delete logs[id.slice(PREFIX.length)];
    }
    surfaces = next;
  }
  // Also recover stale cards after failures that emitted no closing envelope.
  for (const id of Object.keys(surfaces)) {
    if (id.startsWith(PREFIX) && (stopped || id !== activeId)) remember(id);
  }
  return { surfaces, archived: logs };
}
