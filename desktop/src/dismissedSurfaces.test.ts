import { describe, expect, it } from "vitest";
import type { A2uiSurface } from "./a2ui";
import {
  DISMISSED_SCHEDULED_SURFACES_KEY,
  MAX_DISMISSED_SCHEDULED_SURFACES,
  readDismissedScheduledSurfaces,
  rememberDismissedScheduledSurface,
  withoutDismissedSurfaces,
  withoutSurface,
} from "./dismissedSurfaces";

function memoryStorage(initial = "") {
  let value = initial;
  return {
    getItem: (key: string) => key === DISMISSED_SCHEDULED_SURFACES_KEY ? value : null,
    setItem: (key: string, next: string) => { if (key === DISMISSED_SCHEDULED_SURFACES_KEY) value = next; },
    value: () => value,
  };
}

function surface(surfaceId: string): A2uiSurface {
  return { surfaceId, catalogId: "alphonse.desktop.catalog.v1", components: {}, dataModel: {} };
}

describe("dismissed scheduled-task surfaces", () => {
  it("removes only acknowledged cards and keeps independent cards available", () => {
    const surfaces = {
      "scheduled-task:one": surface("scheduled-task:one"),
      "scheduled-task:two": surface("scheduled-task:two"),
      "question:q1": surface("question:q1"),
    };

    const filtered = withoutDismissedSurfaces(surfaces, new Set(["scheduled-task:one"]));

    expect(Object.keys(filtered)).toEqual(["scheduled-task:two", "question:q1"]);
    expect(Object.keys(withoutSurface(filtered, "scheduled-task:two"))).toEqual(["question:q1"]);
  });

  it("persists valid scheduled cards and bounds replay acknowledgements", () => {
    const storage = memoryStorage();
    let dismissed = new Set<string>();
    for (let index = 0; index < MAX_DISMISSED_SCHEDULED_SURFACES + 5; index += 1) {
      dismissed = rememberDismissedScheduledSurface(storage, dismissed, `scheduled-task:${index}`);
    }
    dismissed = rememberDismissedScheduledSurface(storage, dismissed, "question:not-accepted");

    const restored = readDismissedScheduledSurfaces(storage);

    expect(restored.size).toBe(MAX_DISMISSED_SCHEDULED_SURFACES);
    expect(restored.has("scheduled-task:0")).toBe(false);
    expect(restored.has(`scheduled-task:${MAX_DISMISSED_SCHEDULED_SURFACES + 4}`)).toBe(true);
    expect(storage.value()).not.toContain("question:not-accepted");
  });

  it("recovers safely from malformed local storage", () => {
    expect(readDismissedScheduledSurfaces(memoryStorage("not-json"))).toEqual(new Set());
  });
});
